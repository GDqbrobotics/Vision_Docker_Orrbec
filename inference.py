import argparse 
import numpy as np
import cv2
import time
import json
import paho.mqtt.client as mqtt
from multiprocessing import Process, Queue
from PIL import Image
from PIL import ImageOps
from ultralytics import YOLO
import pyorbbecsdk
from examples.utils import frame_to_bgr_image

_initialized = False

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

# This class is needed to pass the parameters between processes because the original types cannot be pickled
class Parameters:
    def __init__(self, _depth_intrinsics, _extrinsic):
        self.fx = _depth_intrinsics.fx
        self.fy = _depth_intrinsics.fy
        self.cx = _depth_intrinsics.cx
        self.cy = _depth_intrinsics.cy
        self.width = _depth_intrinsics.width
        self.height = _depth_intrinsics.height
        self.rot = _extrinsic.rot
        self.transform = _extrinsic.transform


def get_frame_data(color_frame, depth_frame):

    global _initialized

    color_frame = color_frame.as_video_frame()
    depth_frame = depth_frame.as_video_frame()

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    color_width = color_frame.get_width()
    color_height = color_frame.get_height()

    color_profile = color_frame.get_stream_profile()
    depth_profile = depth_frame.get_stream_profile()
    print("video profile:", color_profile.as_video_stream_profile())
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
    color_distortion = color_profile.as_video_stream_profile().get_distortion()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
    depth_distortion = depth_profile.as_video_stream_profile().get_distortion()

    print("depth intrinsics:", depth_intrinsics)

    extrinsic = depth_profile.get_extrinsic_to(color_profile)

    print("extrinsic:", extrinsic)
    _initialized = True
    return color_width, color_height, depth_width, depth_height, color_intrinsics, color_distortion, depth_intrinsics, depth_distortion, extrinsic


def transform_points(x, y, depth, depth_intrinsics, extrinsic):
    res = pyorbbecsdk.transformation2dto3d(pyorbbecsdk.OBPoint2f(x, y), depth, depth_intrinsics, extrinsic)
    original_point = (x , y , depth)
    print(f"\n--- Point Transformation ---")
    print(f"Original point: {original_point}")
    print("Transformed point:",res)
    print(f"--------------------------------------------")
    return res.z, res.x, res.y

def read_camera(*, frame_queue, stream_queue, parameters_queue,  width, height, verbose=False):
    # Create a pipeline with default device
    pipeline = pyorbbecsdk.Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    config = pyorbbecsdk.Config()  # Initialize the config for the pipeline
    
    try:
        # Enable depth and color sensors
        for sensor_type in [pyorbbecsdk.OBSensorType.DEPTH_SENSOR, pyorbbecsdk.OBSensorType.COLOR_SENSOR]:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            assert profile_list is not None
            profile = profile_list.get_default_video_stream_profile()
            assert profile is not None
            print(f"{sensor_type} profile:", profile)
            config.enable_stream(profile)  # Enable the stream for the sensor
    except Exception as e:
        print(e)
        return

    print("start pipeline")
    pipeline.start(config)  # Start the pipeline with the config

    while True:
        # Wait for frames from the pipeline (with a timeout of 100 ms)
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        # Get depth and color frames from the captured frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Skip iteration if depth or color frame is not available
        if depth_frame is None or color_frame is None:
            continue

        if verbose: print("[STREAM] Read rgb frame of size", color_image.shape)
        if verbose: print("[STREAM] Read depth frame of size", depth_image.shape)

        if not _initialized: 
            _color_width, _color_height, _depth_width, _depth_height, _color_intrinsics, _color_distortion, _depth_intrinsics, _depth_distortion, _extrinsic = get_frame_data(color_frame, depth_frame)
            parameters_queue.put(Parameters(_depth_intrinsics, _extrinsic))
                
        # the depth frame has lower resolution than the color frame, so we need to resize it
        # to match the size of the color frame. We use the nearest neighbor interpolation
        # to avoid creating new data points (which could lead to incorrect depth values)
        depth_data = cv2.resize(
            np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(_depth_height, _depth_width),
            (_color_width, _color_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        # Apply temporal filtering
        depth = temporal_filter.process(depth_data)
        
        image = frame_to_bgr_image(color_frame)

        if not frame_queue.full():
            frame_queue.put((image, depth))


def send(*, topic, topic_single, host, port, send_queue, username="", password="", verbose=False):
    def on_connect(client, userdata, flags, rc):
        print("[SEND] Connected to MQTT broker with result code "+str(rc))

    client = mqtt.Client()
    if len(username) > 0 or len(password) > 0:
        print("Setting username and password")
        client.username_pw_set(username, password)
    
    client.connect(host, port, 60)
    client.on_connect = on_connect
    client.loop_start()

    while True:
        event = send_queue.get()
        if verbose: print("[SEND] Sending event: ", event)
        client.publish(topic, json.dumps(event))
        for tomato in event:
            client.publish(topic_single, json.dumps(tomato))
            if verbose: print("[SEND] Sending event signle: ", tomato)

def inference(*, model, frame_queue, parameters_queue, send_queue, min_confidence=0.45, verbose=False, sleep=0):
    model = YOLO(model)
    print("[INFERENCE] Loaded model")
    
    while True:
        image, depth = frame_queue.get()
        if not parameters_queue.empty(): 
            parameters = parameters_queue.get()
            depth_intrinsics = pyorbbecsdk.OBCameraIntrinsic()
            extrinsic = pyorbbecsdk.OBExtrinsic()
            depth_intrinsics.fx = parameters.fx
            depth_intrinsics.fy = parameters.fy
            depth_intrinsics.cx = parameters.cx
            depth_intrinsics.cy = parameters.cy
            depth_intrinsics.width = parameters.width
            depth_intrinsics.height = parameters.height
            extrinsic.rot = parameters.rot
            extrinsic.transform = parameters.transform


        if verbose: print("[INFERENCE] Inference on image of size", image.size)

        results = model.predict(image, stream=True, conf=min_confidence, show=True, verbose=False)
        
        message = parse(results, depth, depth_intrinsics, extrinsic)

        if len(message) > 0:
            send_queue.put(message)
        
        if sleep > 0:
            time.sleep(sleep)

    cv2.destroyAllWindows()

def parse(results,depth, depth_intrinsics, extrinsic):    
    message = []

    for result in results:
        i = 0
        result_json = json.loads(result.to_json())
                
        for object_res in result_json:
            if object_res["name"] == "Talea":
                keypoints = object_res["keypoints"]
                t_bottom_x = keypoints["x"][0]
                t_bottom_y = keypoints["y"][0]
                t_top_x = keypoints["x"][1]
                t_top_y = keypoints["y"][1]
                t_middle_x = keypoints["x"][2]
                t_middle_y = keypoints["y"][2]

                t_bottom_z = 0
                t_top_z = 0
                t_middle_z = 0

                try:
                    t_bottom_z = depth[int(t_bottom_y), int(t_bottom_x)].item()
                    t_top_z = depth[int(t_top_y), int(t_top_x)].item()
                    t_middle_z = depth[int(t_middle_y), int(t_middle_x)].item()
                except Exception as e:
                    print(e)
                    continue
                
                if t_bottom_z == 0 or t_top_z == 0 or t_middle_z == 0:
                    continue

                bottom_z, bottom_x, bottom_y = transform_points(t_bottom_x, t_bottom_y, t_bottom_z, depth_intrinsics, extrinsic)
                top_z, top_x, top_y = transform_points(t_top_x, t_top_y, t_top_z, depth_intrinsics, extrinsic)
                middle_z, middle_x, middle_y = transform_points(t_middle_x, t_middle_y, t_middle_z, depth_intrinsics, extrinsic)

                if t_bottom_z*t_top_z*t_middle_z != 0: #check for possible occlusion on the depth image
                    message.append({
                        "Talea number": i,
                        "X_bottom": bottom_x,
                        "Y_bottom": bottom_y,
                        "Z_bottom": bottom_z,
                        "X_top": top_x,
                        "Y_top": top_y,
                        "Z_top": top_z,
                        "X_middle": middle_x,
                        "Y_middle": middle_y,
                        "Z_middle": middle_z,
                        "confidence": object_res["confidence"]
                    })
            
            i += 1

    return message

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="app/best.pt",
        help="Path to the YOLOv11 model"
    )

    parser.add_argument(
        "--stream-width",
        type=int,
        default=1280,
        help="Width of the stream"
    )
    parser.add_argument(
        "--stream-height",
        type=int,
        default=720,
        help="Height of the stream"
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="192.168.139.70",
        help="Host of the MQTT broker"
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="Port of the MQTT broker"
    )
    parser.add_argument(
        "--mqtt-user",
        type=str,
        default="mqtt",
        help="MQTT username"
    )
    parser.add_argument(
        "--mqtt-password",
        type=str,
        default="Vn370gi@lo#T",
        help="MQTT password"
    )
    parser.add_argument(
        "--mqtt-send-topic",
        type=str,
        default="test_coordinate",
        help="MQTT topic to publish events to"
    )
    parser.add_argument(
        "--mqtt-send-topic-single",
        type=str,
        default="test_coordinate_single",
        help="MQTT topic to publish events to (single)"
    )
    parser.add_argument(
        "--inference-sleep",
        type=float,
        default=0.01,
        help="Sleep time between inferences"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Confidence threshold for object detection"
    )

    args = parser.parse_args()

    frame_queue = Queue(maxsize=1)
    stream_queue = Queue(maxsize=1)
    send_queue = Queue()
    parameters_queue = Queue(maxsize=1)

    read_process = Process(
        target=read_camera,
        kwargs=dict(
            width=args.stream_width,
            height=args.stream_height,
            frame_queue=frame_queue,
            stream_queue=stream_queue,
            parameters_queue=parameters_queue,
            verbose=args.verbose,
        )
    )

    send_process = Process(
        target=send,
        kwargs=dict(
            topic=args.mqtt_send_topic,
            topic_single=args.mqtt_send_topic_single,
            host=args.mqtt_host,
            port=args.mqtt_port,
            send_queue=send_queue,
            username=args.mqtt_user,
            password=args.mqtt_password,
            verbose=args.verbose,
        )
    )

    inference_process = Process(
        target=inference,
        kwargs=dict(
            model=args.model,
            parameters_queue=parameters_queue,
            frame_queue=frame_queue,
            send_queue=send_queue,
            min_confidence=args.min_confidence,
            verbose=args.verbose,
            sleep=args.inference_sleep,
        )
    )

    read_process.start()
    send_process.start()
    inference_process.start()

    inference_process.join()

    send_process.terminate()
    read_process.terminate()


if __name__ == "__main__":
    main()
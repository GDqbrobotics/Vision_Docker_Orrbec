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
from typing import Union, Any, Optional
from pyorbbecsdk import *
import matplotlib.pyplot as plt
import math

_initialized = False

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

CROP_WIDTH = 1400
CROP_HEIGHT = 1080
CROP_STARTING_ROW = int((1080 - int(CROP_HEIGHT))/2)
CROP_STARTING_COL = int((1920 - int(CROP_WIDTH))/2)

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

class ImageCropper:
    def __init__(self, width, height, starting_row, starting_col):
        self.width = width
        self.height = height
        self.starting_row = starting_row
        self.starting_col = starting_col

    def crop(self, image):
        end_row, end_col = self.starting_row + self.height, self.starting_col + self.width
        return image[self.starting_row:end_row, self.starting_col:end_col, :]
    
    def cropped2orig(self, row, col):
        return row + self.starting_row, col + self.starting_col

# This class is needed to pass the parameters between processes because the original types cannot be pickled
class Parameters:
    def __init__(self, _depth_intrinsics, _extrinsic, _color_width, _color_height):
        self.fx = _depth_intrinsics.fx
        self.fy = _depth_intrinsics.fy
        self.cx = _depth_intrinsics.cx
        self.cy = _depth_intrinsics.cy
        self.width = _depth_intrinsics.width
        self.height = _depth_intrinsics.height
        self.rot = _extrinsic.rot
        self.transform = _extrinsic.transform
        self.color_width = _color_width
        self.color_height = _color_height

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

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
    res = transformation2dto3d(OBPoint2f(x, y), depth, depth_intrinsics, extrinsic)
    original_point = (x , y , depth)
    # print(f"\n--- Point Transformation ---")
    # print(f"Original point: {original_point}")
    # print("Transformed point:",res)
    # print(f"--------------------------------------------")
    return res.z, res.x, res.y

def read_camera(*, frame_queue, parameters_queue,  width, height, verbose=False):
    # Create a pipeline with default device
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    config = Config()  # Initialize the config for the pipeline
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)    
    try:
        # Enable depth and color sensors
        for sensor_type in [OBSensorType.DEPTH_SENSOR, OBSensorType.COLOR_SENSOR]:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            assert profile_list is not None
            profile = profile_list.get_default_video_stream_profile()
            try:
                for profile_iterator in profile_list:
                    if profile_iterator.get_width() == width and profile_iterator.get_height() == height:
                        profile = profile_iterator
                        break
            except Exception as e:
                print(e)
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
        if not frames:
            continue

        # --- Spatial Alignment ---
        # Transforms one stream to the coordinate system/FOV of the other
        frames = align_filter.process(frames)
        if not frames:
            continue
        
        frames = frames.as_frame_set()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Skip iteration if depth or color frame is not available
        if depth_frame is None or color_frame is None:
            continue

        if verbose: print("[STREAM] Read rgb frame of size", color_image.shape)
        if verbose: print("[STREAM] Read depth frame of size", depth_image.shape)

        if not _initialized: 
            _color_width, _color_height, _depth_width, _depth_height, _color_intrinsics, _color_distortion, _depth_intrinsics, _depth_distortion, _extrinsic = get_frame_data(color_frame, depth_frame)
            parameters_queue.put(Parameters(_depth_intrinsics, _extrinsic, _color_width, _color_height))
                
        # the depth frame has lower resolution than the color frame, so we need to resize it
        # to match the size of the color frame. We use the nearest neighbor interpolation
        # to avoid creating new data points (which could lead to incorrect depth values)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(_depth_height, _depth_width)
        
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

def inference(*, model, frame_queue, parameters_queue, send_queue, min_confidence=0.45, verbose=True, sleep=0):
    model = YOLO(model)
    print("[INFERENCE] Loaded model")
    
    while True:
        image, depth = frame_queue.get()
        if not parameters_queue.empty(): 
            parameters = parameters_queue.get()
            depth_intrinsics = OBCameraIntrinsic()
            extrinsic = OBExtrinsic()
            depth_intrinsics.fx = parameters.fx
            depth_intrinsics.fy = parameters.fy
            depth_intrinsics.cx = parameters.cx
            depth_intrinsics.cy = parameters.cy
            depth_intrinsics.width = parameters.width
            depth_intrinsics.height = parameters.height
            extrinsic.rot = parameters.rot
            extrinsic.transform = parameters.transform
            color_width = parameters.color_width
            color_height = parameters.color_height

        if color_height*color_width == 0:
            continue

        coeff_height = depth_intrinsics.height / color_height
        coeff_width = depth_intrinsics.width / color_width

        if verbose: print("[INFERENCE] Inference on image of size", image.size)
        

        cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
        image = cropper.crop(image)

        results = list(model.predict(image, stream=True, conf=min_confidence, show=False, verbose=False))
        
        for i, r in enumerate(results):
            # Save results to disk
            r.save(filename=f"result.jpg")
        closure_factor = None
        box = []
        
        closure_factor, box, wrist, knuckle_ref = process_hand_results(results)
        message = parse_box(box, depth, depth_intrinsics, extrinsic, coeff_height, coeff_width)
        if len(message) > 0:
            orientation = wrist - knuckle_ref
            message[0]["closure_factor"] = closure_factor
            message[0]["yaw"] = math.atan2(orientation[1], orientation[0])  # radians
            send_queue.put(message)
            # print("[INFERENCE] Sent message: ", message)
        
        if sleep > 0:
            time.sleep(sleep)

    cv2.destroyAllWindows()

def process_hand_results(results, verbose=False):
    closure_factor = None
    box = []
    wrist = None
    knuckle_ref = None
    for result in results:
        # 1. Get Bounding Boxes in original pixel coordinates (xyxy)
        # boxes.xyxy returns [xmin, ymin, xmax, ymax]
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) != 1:
            continue
        # 2. Get Keypoints (assuming 21 hand keypoints standard)
        # keypoints.xy returns [N, 21, 2]
        if result.keypoints is not None:
            kpts_all = result.keypoints.xy.cpu().numpy()
            
            for i, box in enumerate(boxes):
                if verbose: print(f"Hand {i} Bounding Box (px): {box}")
                
                # Get keypoints for this specific hand
                kpts = kpts_all[i] 
                
                # Hand Closure Logic:
                # Keypoint indices (standard MediaPipe/YOLO hand format):
                # 0: Wrist, 4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
                # 9: Middle finger knuckle (MCP) - used as a reference for hand scale
                
                wrist = kpts[0]
                fingertips = [kpts[8], kpts[12], kpts[16], kpts[20]] # Index, Middle, Ring, Pinky
                knuckle_ref = kpts[9] # Middle finger knuckle
                
                # Calculate reference length (Wrist to Middle Knuckle) to handle zoom/distance
                hand_scale = np.linalg.norm(wrist - knuckle_ref)
                
                if hand_scale == 0: continue # Avoid division by zero
                
                # Calculate average distance of fingers to wrist
                # When open, tips are far from wrist. When closed (fist), tips are near knuckles/wrist.
                distances = [np.linalg.norm(tip - wrist) for tip in fingertips]
                avg_dist = np.mean(distances)
                
                # Normalized factor calculation:
                # A common heuristic: 
                # Fully open: avg_dist is ~2.5x the hand_scale
                # Fully closed: avg_dist is ~1.0x the hand_scale
                # We map this to 0 (open) to 1 (closed)
                closure_raw = (2.5 - (avg_dist / hand_scale)) / 1.5
                closure_factor = np.clip((closure_raw-0.9)*8, 0, 1)
                
                if verbose: print(f"Hand {i} Closure Factor: {closure_factor:.2f} (1.0 = Fist, 0.0 = Open)")
    return closure_factor, box, wrist, knuckle_ref

def parse(wrist, knuckle_ref, depth, depth_intrinsics, extrinsic, coeff_height, coeff_width):
    message = []
    if wrist is None or knuckle_ref is None:
        return message
    print("Wrist:", wrist)
    print("Knuckle Reference:", knuckle_ref)
    cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
    wrist[1], wrist[0] = cropper.cropped2orig(wrist[1], wrist[0])
    knuckle_ref[1], knuckle_ref[0] = cropper.cropped2orig(knuckle_ref[1], knuckle_ref[0])
    wrist[0] = wrist[0] * coeff_width
    wrist[1] = wrist[1] * coeff_height
    knuckle_ref[0] = knuckle_ref[0] * coeff_width
    knuckle_ref[1] = knuckle_ref[1] * coeff_height
    wrist_depth = depth[int(wrist[0]), int(wrist[1])]
    knuckle_ref_depth = depth[int(knuckle_ref[0]), int(knuckle_ref[1])]
    if wrist_depth == 0 or knuckle_ref_depth == 0:
        print("Invalid depth at wrist or knuckle reference point.")
        return message
    
    knuckle_z, knuckle_x, knuckle_y = transform_points(knuckle_ref[0], knuckle_ref[1], knuckle_ref_depth, depth_intrinsics, extrinsic)
    wrist_z, wrist_x, wrist_y = transform_points(wrist[0], wrist[1], wrist_depth, depth_intrinsics, extrinsic)
    message.append({
        "X_centroid": (wrist_x+knuckle_x)/2,
        "Y_centroid": (wrist_y+knuckle_y)/2,
        "Z_centroid": (wrist_z+knuckle_z)/2,
        "orientation": {
            "wrist_to_knuckle_x": wrist_x - knuckle_x,
            "wrist_to_knuckle_y": wrist_y - knuckle_y,
            "wrist_to_knuckle_z": wrist_z - knuckle_z
        }
    })
    return message

def parse_box(box,depth, depth_intrinsics, extrinsic, coeff_height, coeff_width):    
    message = []
    if len(box) != 4:
        return message
    
    cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
    box[1], box[0] = cropper.cropped2orig(box[1], box[0])
    box[3], box[2] = cropper.cropped2orig(box[3],box[2])
    # print(box)
    target_x_min = int(box[0]*coeff_width)
    target_y_min = int(box[1]*coeff_height)
    target_x_max = int(box[2]*coeff_width)
    target_y_max = int(box[3]*coeff_height)
    target_z = 0

    try:
        x_array = range(target_x_min, target_x_max, 1) #width or col
        y_array = range(target_y_min, target_y_max, 1) #height or row
        X, Y = np.meshgrid(x_array, y_array)
        
        Z = depth[Y, X] # is a matrix so the order is Y,X (row,col) and not X,Y
        Z_max = np.max(Z) 
        Z = np.where(Z > 200, Z, Z_max)  # Replace outliers with max
        # Remove isolated spikes by applying a median filter
        Z = cv2.medianBlur(Z, 5)  # Kernel size of 5
        Z = cv2.medianBlur(Z, 5)
        Z = cv2.medianBlur(Z, 5)

        # Calculate the middle value between the median and the lowest point

        Z_nonzero = Z.flatten()
        if len(Z_nonzero) == 0:
            print("No valid depth points found.")
            return message
        Z_min = np.min(Z_nonzero)
        Z_max = np.max(Z_nonzero)
        middle_value = (Z_max + Z_min) / 2

        # Filter points lower than the middle value
        mask = Z < middle_value
        X_filtered = X[mask]
        Y_filtered = Y[mask]
        Z_filtered = Z[mask]

        if len(Z_filtered) == 0:
            print("No points below the middle value.")
            return message

        # Calculate the geometric centroid
        centroid_x = float(np.mean(X_filtered))
        centroid_y = float(np.mean(Y_filtered))
        centroid_z = float(np.mean(Z_filtered))
        
        final_target_z, final_target_x, final_target_y = transform_points(centroid_x, centroid_y, centroid_z, depth_intrinsics, extrinsic)
        
        message.append({
            "X_centroid": final_target_x,
            "Y_centroid": final_target_y,
            "Z_centroid": final_target_z,
        })

        if False:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(X,Y,Z, c=Z, cmap='viridis')
            # ax.scatter3D(X_filtered,Y_filtered,Z_filtered, c=Z_filtered, cmap='viridis')
            plt.show()

    except Exception as e:
        print(e)
        return message

    return message

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="hand_pose2.pt",
        help="Path to the YOLOv11 model"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Width of the input image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Height of the input image"
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="192.168.139.80",
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
        default=0.2,
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
        default=0.3,
        help="Confidence threshold for object detection"
    )

    args = parser.parse_args()

    frame_queue = Queue(maxsize=1)
    send_queue = Queue()
    parameters_queue = Queue(maxsize=1)

    read_process = Process(
        target=read_camera,
        kwargs=dict(
            width=args.width,
            height=args.height,
            frame_queue=frame_queue,
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
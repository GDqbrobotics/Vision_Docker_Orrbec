import argparse 
import numpy as np
import cv2
import time
import json
import paho.mqtt.client as mqtt
from multiprocessing import Process, Queue
from PIL import Image
from PIL import ImageOps
# from ultralytics import YOLO
import pyorbbecsdk
from examples.utils import frame_to_bgr_image
from ollama import chat 
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from vosk import Model, KaldiRecognizer, SetLogLevel

_initialized = False

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

CROP_WIDTH = 1000
CROP_HEIGHT = 1000
CROP_STARTING_ROW = int((1080 - int(CROP_HEIGHT))/2)
CROP_STARTING_COL = int((1920 - int(CROP_WIDTH))/2)


def record_audio(prompt_queue):
    # Sampling frequency
    freq = 44100

    # Recording duration
    duration = 6

    sd.default.device = [24,24]
    device_list=sd.query_devices()
    print(device_list)

    model = Model(lang="en-us")

    # You can also init model by name or with a folder path
    # model = Model(model_name="vosk-model-en-us-0.21")
    # model = Model("models/en")

    rec = KaldiRecognizer(model, freq)
    rec.SetWords(True)
    rec.SetPartialWords    
    
    while True:
        print("Start recording")
        # Start recorder with the given values 
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq), dtype="int16", samplerate=freq, channels=1)

        # Record audio for the given number of seconds
        sd.wait()
        print("Stop recording")

        data = bytes(recording)

        if rec.AcceptWaveform(data):
            result_json = json.loads(rec.Result())
            print(result_json)
            prompt_queue.put(result_json["text"])


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
    # print(f"\n--- Point Transformation ---")
    # print(f"Original point: {original_point}")
    # print("Transformed point:",res)
    # print(f"--------------------------------------------")
    return res.z, res.x, res.y

def read_camera(*, frame_queue, parameters_queue,  width, height, verbose=False):
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

def inference(*, model, frame_queue, parameters_queue, send_queue, prompt_queue, min_confidence=0.45, verbose=True, sleep=0):
    # model = YOLO(model)
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
            color_width = parameters.color_width
            color_height = parameters.color_height

        if color_height*color_width == 0:
            continue
        if prompt_queue.empty():
            continue

        coeff_height = depth_intrinsics.height / color_height
        coeff_width = depth_intrinsics.width / color_width

        if verbose: print("[INFERENCE] Inference on image of size", image.size)
        

        cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
        image = cropper.crop(image)
        
        filename=f"app/result.jpg"
        ima = Image.fromarray(image[:, :, ::-1])
        ima.save(filename)
        prompt = prompt_queue.get()
        print(prompt)
        # results = model.predict(image, stream=True, conf=min_confidence, show=False, verbose=False)
        response = chat(
            model='gemma4',
            messages=[
                {
                    'role': 'system',
                    'content': f'You are the vision system of a robotic arm. Your duty is to provide the pixel coordinates in format COORDINATES[x_pixel,y_pixel]. Remember: the picture size is {CROP_WIDTH}x{CROP_HEIGHT} the origin is top left corner, this will be used by the arm to pick an object.'
                },
                {
                'role': 'user',
                'content': prompt,
                'images': [filename],
                }
            ],
            think=False,
            stream=False,
        )
        sentence = response.message.content
        print(sentence)
        red = [0,0,255]
        starter = sentence.find("COORDINATES[")
        message = []
        if starter != -1:
            starter = starter + 12
            comma = sentence[starter:].find(",")
            ender = sentence[starter:].find("]")
            if comma*ender > 0:
                try:
                    y = int(sentence[starter:][:comma])
                    x = int(sentence[starter:][comma+1:ender])
                    message = parse(x,y, depth, depth_intrinsics, extrinsic, coeff_height, coeff_width)
                    image2=image[:, :, ::-1]
                    image2[x:x+20,y:y+20]=red                 
                    Image.fromarray(image2).save(f"app/prova.jpg")
                except Exception as e:
                    print(e)
            
        # for i, r in enumerate(results):
        #     r.save(filename=f"app/result.jpg")

        if len(message) > 0:
            send_queue.put(message)
        
        # if sleep > 0:
        #     time.sleep(sleep)

    cv2.destroyAllWindows()

def parse(x,y,depth, depth_intrinsics, extrinsic, coeff_height, coeff_width):    
    message = []
    cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
    y, x = cropper.cropped2orig(y, x)
    target_x = x*coeff_width
    target_y = y*coeff_height
    target_z = 0

    try:
        target_z = depth[int(target_y), int(target_x)].item()

    except Exception as e:
        print(e)
        return message
    
    if target_z == 0:
        return message

    target_z, target_x, target_y = transform_points(target_x, target_y, target_z, depth_intrinsics, extrinsic)

    if target_z != 0: #check for possible occlusion on the depth image
        message.append({
            "X_target": target_x,
            "Y_target": target_y,
            "Z_target": target_z,
        })

    return message

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="app/yolov8x-oiv7.pt",
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
        default="192.168.139.122",
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
    send_queue = Queue()
    prompt_queue = Queue(maxsize=2)
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

    record_mic_process = Process(
        target=record_audio,
        kwargs=dict(
            prompt_queue=prompt_queue,
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
            prompt_queue=prompt_queue,
            min_confidence=args.min_confidence,
            verbose=args.verbose,
            sleep=args.inference_sleep,
        )
    )

    read_process.start()
    send_process.start()
    record_mic_process.start()
    inference_process.start()

    inference_process.join()

    send_process.terminate()
    read_process.terminate()
    record_mic_process.terminate()

if __name__ == "__main__":
    main()
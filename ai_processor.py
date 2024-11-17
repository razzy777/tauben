import argparse
import os
import sys
from pathlib import Path
import numpy as np
import queue
import threading
from PIL import Image, ImageDraw, ImageFont
from typing import List, Generator, Optional, Tuple, Dict
from functools import partial
from hailo_platform import (
    HEF, VDevice, FormatType, HailoSchedulingAlgorithm
)
import socketio
import base64
import cv2

# Constants
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')


class HailoAsyncInference:
    def __init__(
        self,
        hef_path: str,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(
        self, completion_info, bindings_list: list, input_batch: list,
    ) -> None:
        if completion_info.exception:
            print(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                try:
                    output_data = {}
                    for name in bindings._output_names:
                        output_buffer = bindings.output(name).get_buffer()
                        if isinstance(output_buffer, list):
                            # Concatenate list of arrays into a single array
                            output_buffer = np.concatenate(output_buffer, axis=0)
                        output_data[name] = output_buffer
                    self.output_queue.put((input_batch[i], output_data))
                except Exception as e:
                    print(f"Error in callback processing result {i}: {e}")
                    import traceback
                    traceback.print_exc()

    def get_vstream_info(self) -> Tuple[list, list]:
        return (
            self.hef.get_input_vstream_infos(),
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break  # Sentinel value to stop the inference loop

                if self.send_original_frame:
                    original_batch, preprocessed_batch = batch_data
                else:
                    preprocessed_batch = batch_data

                bindings_list = []
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback,
                        input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                        bindings_list=bindings_list
                    )
                )
            job.wait(10000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            return self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )


def generate_color(class_id: int) -> tuple:
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.label_font = label_font

    def get_labels(self, labels_path: str) -> list:
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """
        Preprocess the image for inference.

        Args:
            image (np.ndarray): Input image in BGR format.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            np.ndarray: Preprocessed image.
        """
        # Resize with aspect ratio maintained
        img_h, img_w = image.shape[:2]
        scale = min(model_w / img_w, model_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        delta_w = model_w - new_w
        delta_h = model_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Apply padding
        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_color
        )

        # Convert BGR to RGB
        padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        padded_image = padded_image.astype(np.float32) / 255.0

        # Transpose to CHW format
        padded_image = np.transpose(padded_image, (2, 0, 1))

        return padded_image

    def draw_detection(self, image: Image.Image, box: list, cls: int, score: float, color: tuple):
        draw = ImageDraw.Draw(image)
        label = f"{self.labels[cls]}: {score:.2f}%"
        xmin, ymin, xmax, ymax = box
        font = ImageFont.truetype(self.label_font, size=15)
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
        draw.text((xmin + 4, ymin + 4), label, fill=color, font=font)

    def visualize(self, detections: dict, image: Image.Image, image_id: int, output_path: str, min_score: float = 0.25):
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                color = generate_color(classes[idx])
                box = boxes[idx]
                self.draw_detection(image, box, classes[idx], scores[idx] * 100.0, color)

        image.save(f'{output_path}/output_image{image_id}.jpg', 'JPEG')

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (dict): Raw detections from the model.
            orig_image_shape (Tuple[int, int]): Original image shape (height, width).

        Returns:
            dict: Filtered detection results.
        """
        boxes = []
        scores = []
        classes = []
        num_detections = 0

        output_tensor = input_data[next(iter(input_data))]  # Get the first (and only) output tensor
        if output_tensor.size == 0:
            return {
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': num_detections
            }

        # Assuming the output tensor is of shape [num_detections, 6]
        # where each detection is [x1, y1, x2, y2, confidence, class_id]
        for det in output_tensor:
            x1, y1, x2, y2, confidence, class_id = det[:6]
            score = float(confidence)
            if score >= 0.1:
                # Scale boxes to original image size
                h, w = orig_image_shape
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                classes.append(int(class_id))
                num_detections += 1

        return {
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
            'num_detections': num_detections
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="/Downloads/yolov8n.hef"
    )
    parser.add_argument(
        "-l", "--labels",
        default="coco.txt",
        help="Path to a text file containing labels."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    hef_path = args.net
    labels_path = args.labels
    batch_size = args.batch_size

    # Initialize utility class
    utils = ObjectDetectionUtils(labels_path)

    # Queues for inference
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Initialize Hailo inference
    hailo_inference = HailoAsyncInference(
        hef_path=hef_path,
        input_queue=input_queue,
        output_queue=output_queue,
        batch_size=batch_size,
        input_type='FLOAT32',  # Assuming model expects float32 inputs
        send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    # Start inference thread
    inference_thread = threading.Thread(target=hailo_inference.run)
    inference_thread.start()

    # Socket.IO client setup
    sio = socketio.Client()

    @sio.event(namespace='/ai')
    def connect():
        print('Connected to AI namespace')

    @sio.event(namespace='/ai')
    def connect_error(data):
        print(f'Connection error: {data}')

    @sio.event(namespace='/ai')
    def disconnect():
        print('Disconnected from AI namespace')

    @sio.on('videoFrame', namespace='/ai')
    def on_video_frame(frame_data):
        try:
            # Decode base64 frame
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error: Could not decode frame")
                return

            # Preprocess frame
            preprocessed_frame = utils.preprocess(frame, width, height)

            # Put frame in input queue
            input_queue.put(( [frame], [preprocessed_frame] ))

            # Get results from output queue
            original_frame, outputs = output_queue.get()

            if outputs:
                # Extract detections
                detections = utils.extract_detections(outputs, frame.shape[:2])

                if detections['num_detections'] > 0:
                    # Convert original frame to PIL image for visualization
                    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Visualize detections
                    utils.visualize(detections, original_image, image_id=0, output_path='output_images')

                    # Emit detections to server
                    sio.emit('aiDetections', detections, namespace='/ai')
                else:
                    print("No detections found.")

        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()

    # Connect to server
    connection_url = 'http://localhost:3000'
    sio.connect(
        connection_url,
        namespaces=['/ai'],
        wait_timeout=10,
        transports=['websocket', 'polling'],
        socketio_path='/socket.io',
    )

    print("Connected successfully")
    print("Waiting for events...")
    sio.wait()

    # Cleanup
    input_queue.put(None)  # Signal inference thread to stop
    inference_thread.join()
    if sio.connected:
        sio.disconnect()


if __name__ == "__main__":
    main()

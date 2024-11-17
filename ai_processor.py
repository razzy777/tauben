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
        print(f"Initialized HailoAsyncInference with input shape: {self.get_input_shape()}")

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(self, completion_info, bindings_list: list, input_batch: list) -> None:
        if completion_info.exception:
            print(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                try:
                    output_data = {}
                    for name in bindings._output_names:
                        output_buffer = bindings.output(name).get_buffer()
                        if isinstance(output_buffer, list):
                            output_buffer = np.concatenate(output_buffer, axis=0)
                        output_data[name] = output_buffer
                    self.output_queue.put((input_batch[i], output_data))
                except Exception as e:
                    print(f"Error in callback processing result {i}: {e}")
                    import traceback
                    traceback.print_exc()

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self) -> None:
        try:
            with self.infer_model.configure() as configured_infer_model:
                while True:
                    try:
                        batch_data = self.input_queue.get()
                        if batch_data is None:
                            break  # Sentinel value to stop the inference loop

                        if self.send_original_frame:
                            original_batch, preprocessed_batch = batch_data
                        else:
                            preprocessed_batch = batch_data

                        bindings_list = []
                        for frame in preprocessed_batch:
                            # Ensure frame is C_CONTIGUOUS
                            frame_contiguous = np.ascontiguousarray(frame, dtype=np.float32)
                            
                            # Create bindings and set buffer
                            bindings = self._create_bindings(configured_infer_model)
                            bindings.input().set_buffer(frame_contiguous)
                            bindings_list.append(bindings)

                        configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                        job = configured_infer_model.run_async(
                            bindings_list, partial(
                                self.callback,
                                input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                                bindings_list=bindings_list
                            )
                        )
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                if 'job' in locals():
                    job.wait(10000)  # Wait for the last job
                    
        except Exception as e:
            print(f"Fatal error in inference loop: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_bindings(self, configured_infer_model) -> object:
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=np.float32,
                    order='C'  # Ensure C-contiguous output buffers
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                    order='C'  # Ensure C-contiguous output buffers
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)


def generate_color(class_id: int) -> tuple:
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114)):
        """Initialize the ObjectDetectionUtils class.
        
        Args:
            labels_path (str): Path to the labels file
            padding_color (tuple): RGB color used for padding (default: (114, 114, 114))
        """
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        print(f"Initialized ObjectDetectionUtils with {len(self.labels)} labels")

    def get_labels(self, labels_path: str) -> list:
        """Load labels from file."""
        try:
            with open(labels_path, 'r', encoding="utf-8") as f:
                class_names = f.read().splitlines()
            return class_names
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            return []

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """Preprocess image for inference.
        
        Args:
            image (np.ndarray): Input image in BGR format
            model_w (int): Model input width
            model_h (int): Model input height
            
        Returns:
            np.ndarray: Preprocessed image in CHW format
        """
        try:
            print(f"Preprocessing image shape: {image.shape} to {model_w}x{model_h}")
            
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
                resized_image, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=self.padding_color
            )

            # Convert BGR to RGB
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] and ensure float32
            padded_image = padded_image.astype(np.float32) / 255.0

            # Transpose to CHW format and ensure C-contiguous
            padded_image = np.ascontiguousarray(
                np.transpose(padded_image, (2, 0, 1)),
                dtype=np.float32
            )

            print(f"Preprocessed image shape: {padded_image.shape}")
            return padded_image

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        """Extract detections from model output.
        
        Args:
            input_data (dict): Model output
            orig_image_shape (tuple): Original image shape (height, width)
            
        Returns:
            dict: Processed detections
        """
        try:
            boxes = []
            scores = []
            classes = []
            num_detections = 0

            output_tensor = input_data[next(iter(input_data))]
            print(f"Processing output tensor shape: {output_tensor.shape}")

            if output_tensor.size == 0:
                return {
                    'detection_boxes': boxes,
                    'detection_classes': classes,
                    'detection_scores': scores,
                    'num_detections': num_detections
                }

            # Process each detection
            for det in output_tensor:
                x1, y1, x2, y2, confidence, class_id = det[:6]
                score = float(confidence)
                
                if score >= 0.25:  # Confidence threshold
                    # Scale boxes to original image size
                    h, w = orig_image_shape
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)

                    boxes.append([y1, x1, y2, x2])  # Convert to [ymin, xmin, ymax, xmax]
                    scores.append(score)
                    classes.append(int(class_id))
                    num_detections += 1

            result = {
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': num_detections
            }
            
            print(f"Found {num_detections} detections")
            return result

        except Exception as e:
            print(f"Error extracting detections: {e}")
            import traceback
            traceback.print_exc()
            return {
                'detection_boxes': [],
                'detection_classes': [],
                'detection_scores': [],
                'num_detections': 0
            }

    def format_detections_for_frontend(self, detections: dict, image_shape: tuple) -> list:
        """Format detections for frontend display.
        
        Args:
            detections (dict): Raw detections
            image_shape (tuple): Image shape (height, width)
            
        Returns:
            list: Formatted detections for frontend
        """
        formatted = []
        h, w = image_shape[:2]
        
        for i in range(detections['num_detections']):
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            formatted.append({
                'box': [
                    float(ymin) / h,  # normalized ymin
                    float(xmin) / w,  # normalized xmin
                    float(ymax) / h,  # normalized ymax
                    float(xmax) / w   # normalized xmax
                ],
                'class': self.labels[detections['detection_classes'][i]],
                'score': float(detections['detection_scores'][i])
            })
            
        return formatted

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="/home/johannes/Downloads/yolov8n.hef"
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

        print(f"Received frame shape: {frame.shape}")

        # Preprocess frame
        try:
            preprocessed_frame = utils.preprocess(frame, width, height)
            print("Frame preprocessed successfully")
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return

        # Put frame in input queue
        try:
            input_queue.put(([frame], [preprocessed_frame]))
            print("Frame queued for inference")
        except Exception as e:
            print(f"Queue error: {e}")
            return

        # Get results from output queue with timeout
        try:
            original_frame, outputs = output_queue.get(timeout=2.0)
            print("Got inference results")

            if outputs:
                # Extract detections
                detections = utils.extract_detections(outputs, frame.shape[:2])

                if detections['num_detections'] > 0:
                    # Format detections for frontend
                    formatted_detections = utils.format_detections_for_frontend(detections, frame.shape)
                    
                    # Emit detections to server
                    sio.emit('aiDetections', formatted_detections, namespace='/ai')
                    print(f"Emitted {len(formatted_detections)} detections")
                else:
                    print("No detections found")
            else:
                print("No outputs from model")

        except queue.Empty:
            print("Timeout waiting for inference results")
        except Exception as e:
            print(f"Error processing inference results: {e}")

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

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import queue
import threading
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Generator, Optional, Tuple, Dict
from functools import partial
import socketio
import base64
import cv2
import time
from hailo_platform import (
    HEF, VDevice, FormatType, HailoSchedulingAlgorithm
)

# Constants
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')

class AIClient:
    def __init__(self, server_url='http://localhost:3000', retry_interval=5):
        self.server_url = server_url
        self.retry_interval = retry_interval
        self.sio = socketio.Client()
        self.connected = False
        self.setup_logging()
        self.setup_socket_handlers()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AIClient')
    
    def setup_socket_handlers(self):
        @self.sio.event(namespace='/ai')
        def connect():
            self.connected = True
            self.logger.info('Connected to AI namespace')

        @self.sio.event(namespace='/ai')
        def connect_error(data):
            self.connected = False
            self.logger.error(f'Connection error: {data}')

        @self.sio.event(namespace='/ai')
        def disconnect():
            self.connected = False
            self.logger.info('Disconnected from AI namespace')

    def connect_with_retry(self):
        while not self.connected:
            try:
                self.logger.info(f'Attempting to connect to {self.server_url}')
                self.sio.connect(self.server_url, namespaces=['/ai'])
                break
            except Exception as e:
                self.logger.error(f'Connection failed: {e}')
                self.logger.info(f'Retrying in {self.retry_interval} seconds...')
                time.sleep(self.retry_interval)

    def decode_frame(self, frame_data):
        try:
            # Decode base64 frame
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode frame")
                
            # Save frame for debugging (temporarily)
            cv2.imwrite('last_received_frame.jpg', frame)
            
            self.logger.info(f'Successfully decoded frame: {frame.shape}')
            return frame
            
        except Exception as e:
            self.logger.error(f'Error decoding frame: {e}')
            return None

    def send_detections(self, detections):
        if self.connected:
            try:
                self.logger.info(f'Sending detections: {detections}')
                self.sio.emit('aiDetections', detections, namespace='/ai')
            except Exception as e:
                self.logger.error(f'Error sending detections: {e}')

    def disconnect(self):
        if self.connected:
            self.sio.disconnect()
            self.connected = False

    def is_connected(self):
        return self.connected

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
        self.logger = logging.getLogger('HailoInference')
        
        try:
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
            
            self.logger.info(f'Initialized HailoInference with HEF: {hef_path}')
            
        except Exception as e:
            self.logger.error(f'Error initializing HailoInference: {e}')
            raise

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(self, completion_info, bindings_list: list, input_batch: list) -> None:
        if completion_info.exception:
            self.logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                try:
                    output_data = {}
                    for name in bindings._output_names:
                        output_buffer = bindings.output(name).get_buffer()
                        if isinstance(output_buffer, list):
                            output_buffer = np.concatenate(output_buffer, axis=0)
                        self.logger.debug(f"Output '{name}' shape: {output_buffer.shape}, dtype: {output_buffer.dtype}")
                        self.logger.debug(f"Output '{name}' sample: {output_buffer.ravel()[:10]}")
                        output_data[name] = output_buffer
                    self.output_queue.put((input_batch[i], output_data))
                except Exception as e:
                    self.logger.error(f"Error in callback processing result {i}: {e}")
                    import traceback
                    traceback.print_exc()

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self) -> None:
        with self.infer_model.configure() as configured_infer_model:
            while True:
                try:
                    batch_data = self.input_queue.get(timeout=1.0)
                    if batch_data is None:
                        break  # Sentinel value to stop the inference loop

                    if self.send_original_frame:
                        original_batch, preprocessed_batch = batch_data
                    else:
                        preprocessed_batch = batch_data

                    bindings_list = []
                    for frame in preprocessed_batch:
                        bindings = self._create_bindings(configured_infer_model)
                        bindings.input().set_buffer(np.ascontiguousarray(frame))
                        bindings_list.append(bindings)

                    configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_infer_model.run_async(
                        bindings_list, partial(
                            self.callback,
                            input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                            bindings_list=bindings_list
                        )
                    )
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in inference loop: {e}")
                    import traceback
                    traceback.print_exc()
            
            if 'job' in locals():
                job.wait(10000)  # Wait for the last job

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
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            return self.output_type[output_info.name].lower()

class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), confidence_threshold: float = 0.25):
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger('ObjectDetection')

    def get_labels(self, labels_path: str) -> list:
        try:
            with open(labels_path, 'r', encoding="utf-8") as f:
                class_names = f.read().splitlines()
            self.logger.info(f'Loaded {len(class_names)} labels from {labels_path}')
            return class_names
        except Exception as e:
            self.logger.error(f'Error loading labels: {e}')
            raise

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        try:
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

            self.logger.debug(f'Preprocessed image shape: {padded_image.shape}')
            return padded_image

        except Exception as e:
            self.logger.error(f'Error in preprocessing: {e}')
            raise

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        try:
            boxes = []
            scores = []
            classes = []
            num_detections = 0

            output_tensor = input_data[next(iter(input_data))]
            self.logger.debug(f'Raw output tensor shape: {output_tensor.shape}')

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
                
                if score >= self.confidence_threshold:
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

            self.logger.debug(f'Extracted {num_detections} detections')
            return {
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': num_detections
            }

        except Exception as e:
            self.logger.error(f'Error extracting detections: {e}')
            raise

    def visualize(self, detections: dict, image: Image.Image, image_id: int, output_path: str):
        try:
            os.makedirs(output_path, exist_ok=True)
            
            boxes = detections['detection_boxes']
            classes = detections['detection_classes']
            scores = detections['detection_scores']

            draw = ImageDraw.Draw(image)
            
            for idx in range(detections['num_detections']):
                if scores[idx] >= self.confidence_threshold:
                    color = tuple(np.random.randint(0, 255, size=3).tolist())
                    box = boxes[idx]
                    label = f"{self.labels[classes[idx]]}: {scores[idx]*100:.2f}%"
                    
                    # Draw box
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
                    
                    # Draw label
                    text_bbox = draw.textbbox((box[0], box[1]), label)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=color)
                    draw.text((box[0], box[1]), label, fill='white')

            output_file = os.path.join(output_path, f'output_image{image_id}.jpg')
            image.save(output_file, 'JPEG')
            self.logger.info(f'Saved visualization to {output_file}')

        except Exception as e:
            self.logger.error(f'Error in visualization: {e}')
            raise

def parse_args() -> argparse.Namespace:
    parser = arg
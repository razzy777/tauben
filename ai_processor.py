import argparse
import os
import sys
import numpy as np
import queue
import threading
import socketio
import base64
import cv2
import time
import logging
from typing import Optional, Tuple, Dict
from functools import partial
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

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

        # Always set input to UINT8 first, then convert
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def preprocess_for_hailo(self, frame: np.ndarray) -> np.ndarray:
        """Convert float32 [0,1] to uint8 [0,255]"""
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)
        return frame

    def run(self) -> None:
        try:
            with self.infer_model.configure() as configured_infer_model:
                while True:
                    batch_data = self.input_queue.get()
                    if batch_data is None:
                        break

                    if self.send_original_frame:
                        original_batch, preprocessed_batch = batch_data
                    else:
                        preprocessed_batch = batch_data

                    bindings_list = []
                    for frame in preprocessed_batch:
                        # Convert to uint8
                        frame_uint8 = self.preprocess_for_hailo(frame)
                        frame_contiguous = np.ascontiguousarray(frame_uint8)
                        
                        bindings = self._create_bindings(configured_infer_model)
                        bindings.input().set_buffer(frame_contiguous)
                        bindings_list.append(bindings)

                    configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                    configured_infer_model.run_async(
                        bindings_list, partial(
                            self.callback,
                            input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                            bindings_list=bindings_list
                        )
                    )

        except Exception as e:
            print(f"Fatal error in inference loop: {e}")
            import traceback
            traceback.print_exc()
            raise

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
                        # No need to reshape; use the output as is
                        output_data[name] = output_buffer
                    self.output_queue.put((input_batch[i], output_data))
                except Exception as e:
                    print(f"Error in callback processing result {i}: {e}")
                    import traceback
                    traceback.print_exc()

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def _create_bindings(self, configured_infer_model) -> object:
        output_buffers = {
            output_info.name: np.empty(
                self.infer_model.output(output_info.name).shape,
                dtype=np.float32,
                order='C'
            )
            for output_info in self.hef.get_output_vstream_infos()
        }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114)):
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        # Find person class index
        try:
            self.person_class = self.labels.index('person')
            print(f"Person class index: {self.person_class}")
        except ValueError:
            self.person_class = 0  # Default to 0 if not found
            print("Warning: 'person' not found in labels, using class index 0")
        self.confidence_threshold = 0.90  # Adjust as needed

    def get_labels(self, labels_path: str) -> list:
        try:
            with open(labels_path, 'r', encoding="utf-8") as f:
                class_names = f.read().splitlines()
            return class_names
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            return []

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """
        Preprocess image for YOLOv8 inference.
        """
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

            padded_image = cv2.copyMakeBorder(
                resized_image, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=self.padding_color
            )

            # Transpose to CHW format
            chw_image = np.transpose(padded_image, (2, 0, 1))
            final_image = np.ascontiguousarray(chw_image)
            return final_image

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        """
        Extract person detections from model output.
        """
        try:
            boxes = []
            scores = []
            classes = []

            # Adjust according to your model's output names
            output_name = 'yolov8n/yolov8_nms_postprocess'
            output_tensor = input_data.get(output_name)
            
            if output_tensor is None or output_tensor.size == 0:
                return self._empty_detection_result()

            # Process each detection
            for detection in output_tensor:
                x1, y1, x2, y2, confidence = detection

                if confidence >= self.confidence_threshold:
                    # Scale to image coordinates
                    h, w = orig_image_shape
                    x1_px = int(x1 * w)
                    y1_px = int(y1 * h)
                    x2_px = int(x2 * w)
                    y2_px = int(y2 * h)

                    # Calculate box dimensions
                    width = x2_px - x1_px
                    height = y2_px - y1_px
                    aspect_ratio = height / width if width > 0 else 0

                    # Aspect ratio thresholds
                    MIN_ASPECT_RATIO = 0.5
                    MAX_ASPECT_RATIO = 3.0

                    if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                        boxes.append([y1_px, x1_px, y2_px, x2_px])
                        scores.append(float(confidence))
                        classes.append(self.person_class)

            num_detections = len(scores)

            # Only keep the detection with the highest confidence
            if num_detections > 0:
                max_conf_idx = np.argmax(scores)
                result = {
                    'detection_boxes': [boxes[max_conf_idx]],
                    'detection_classes': [classes[max_conf_idx]],
                    'detection_scores': [scores[max_conf_idx]],
                    'num_detections': 1
                }
            else:
                result = self._empty_detection_result()

            return result

        except Exception as e:
            print(f"Error extracting detections: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_detection_result()

    def _empty_detection_result(self):
        return {
            'detection_boxes': [],
            'detection_classes': [],
            'detection_scores': [],
            'num_detections': 0
        }

    def format_detections_for_frontend(self, detections: dict, image_shape: tuple) -> list:
        """Format person detections for frontend display."""
        try:
            formatted = []
            h, w = image_shape[:2]
            
            for i in range(detections['num_detections']):
                ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
                confidence = detections['detection_scores'][i]
                
                formatted_detection = {
                    'box': [
                        float(ymin) / h,
                        float(xmin) / w,
                        float(ymax) / h,
                        float(xmax) / w
                    ],
                    'class': 'person',  # Always person
                    'score': float(confidence)
                }
                formatted.append(formatted_detection)
            
            return formatted

        except Exception as e:
            print(f"Error formatting detections: {e}")
            import traceback
            traceback.print_exc()
            return []

class AIProcessor:
    def __init__(self, hef_path: str, labels_path: str, server_url: str = 'http://localhost:3000'):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AIProcessor')
        
        # Initialize components
        self.utils = ObjectDetectionUtils(labels_path)
        
        # Initialize queues
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Initialize Hailo inference
        self.hailo_inference = HailoAsyncInference(
            hef_path=hef_path,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            batch_size=1,
            input_type='FLOAT32',
            send_original_frame=True
        )
        
        # Get model dimensions
        self.height, self.width, _ = self.hailo_inference.get_input_shape()
        
        # Initialize socket client
        self.sio = socketio.Client()
        self.server_url = server_url
        self.connected = False
        
        # Setup socket handlers
        self.setup_socket_handlers()
        
        # Threading control
        self.running = False
        self.inference_thread = None
        
        # For frame rate limiting
        self.last_update_time = 0

    def setup_socket_handlers(self):
        @self.sio.event(namespace='/ai')
        def connect():
            self.connected = True
            self.logger.info('Connected to AI namespace')

        @self.sio.event(namespace='/ai')
        def disconnect():
            self.connected = False
            self.logger.info('Disconnected from AI namespace')

        @self.sio.on('videoFrame', namespace='/ai')
        def on_video_frame(frame_data):
            self.process_frame(frame_data)

    def decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame data into an OpenCV image."""
        try:
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
            else:
                self.logger.error("Error: Frame decoding failed")
                return None
        except Exception as e:
            self.logger.error(f"Frame decoding error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_frame(self, frame_data: str):
        try:
            current_time = time.time()
            if current_time - self.last_update_time < 1:
                return

            frame = self.decode_frame(frame_data)
            if frame is None:
                return

            self.last_update_time = current_time  # Update after successful frame decoding

            preprocessed_frame = self.utils.preprocess(frame, self.width, self.height)

            try:
                self.input_queue.put(([frame], [preprocessed_frame]), block=False)
            except queue.Full:
                self.logger.warning("Input queue full, skipping frame")
                return

            try:
                original_frame, outputs = self.output_queue.get(timeout=2.0)

                if outputs:
                    detections = self.utils.extract_detections(outputs, frame.shape[:2])

                    if detections['num_detections'] > 0:
                        formatted_detections = self.utils.format_detections_for_frontend(
                            detections, frame.shape
                        )
                        if formatted_detections:
                            # Log detection information
                            det = formatted_detections[0]
                            box = det['box']
                            self.logger.info(
                                f"Detection: class={det['class']}, score={det['score']:.2f}, box={box}"
                            )
                            self.sio.emit('aiDetections', formatted_detections, namespace='/ai')
                    else:
                        # Less logs when nothing is detected
                        pass
                else:
                    self.logger.info("No outputs received from the inference model.")

            except queue.Empty:
                self.logger.warning("Inference timeout")
            except Exception as e:
                self.logger.error(f"Inference processing error: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()

    def connect(self):
        while not self.connected:
            try:
                self.logger.info(f"Attempting to connect to {self.server_url}")
                self.sio.connect(
                    self.server_url,
                    namespaces=['/ai'],
                    transports=['websocket']
                )
                break
            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                time.sleep(5)

    def start(self):
        self.running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.hailo_inference.run)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        # Connect to server
        self.connect()
        
        # Main loop
        try:
            while self.running:
                if not self.connected:
                    self.connect()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if self.connected:
            self.sio.disconnect()
        # Send sentinel value to stop inference thread
        self.input_queue.put(None)
        if self.inference_thread:
            self.inference_thread.join()
        self.logger.info("AI Processor stopped")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hailo YOLOv8 Object Detection")
    parser.add_argument(
        "--net",
        help="Path to the HEF file",
        default="/home/johannes/Downloads/yolov8n.hef"
    )
    parser.add_argument(
        "--labels",
        help="Path to the labels file",
        default="coco.txt"
    )
    parser.add_argument(
        "--server",
        help="Server URL",
        default="http://localhost:3000"
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize and start the AI processor
        processor = AIProcessor(
            hef_path=args.net,
            labels_path=args.labels,
            server_url=args.server
        )
        
        print(f"\nStarting AI processor with:")
        print(f"  HEF path: {args.net}")
        print(f"  Labels path: {args.labels}")
        print(f"  Server URL: {args.server}")
        
        processor.start()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

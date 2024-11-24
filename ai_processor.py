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
from typing import Optional, Tuple, Dict, List
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
        
        # Get and verify input shape
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        assert input_vstream_info.shape == (640, 640, 3), f"Unexpected model input shape: {input_vstream_info.shape}"
        print(f"Model configured for input shape {input_vstream_info.shape}")
        
        # Configure model
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        
        self.output_type = output_type
        self.send_original_frame = send_original_frame
        print(f"Available output vstream infos: {[info.name for info in self.hef.get_output_vstream_infos()]}")
        print("\nModel Information:")
        print("Input Streams:")
        for info in self.hef.get_input_vstream_infos():
            print(f"- Name: {info.name}")
            print(f"- Shape: {info.shape}")
            print(f"- Format: {info.format}")
        
        print("\nOutput Streams:")
        for info in self.hef.get_output_vstream_infos():
            print(f"- Name: {info.name}")
            print(f"- Shape: {info.shape}")
            print(f"- Format: {info.format}")
        
        # Get output configuration
        output_info = self.hef.get_output_vstream_infos()[0]
        self.output_shape = output_info.shape
        print(f"\nOutput shape: {self.output_shape}")



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
                        # Ensure frame is contiguous uint8
                        frame_contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
                        
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
                        print(f"\nOutput buffer info:")
                        print(f"- Name: {name}")
                        
                        if isinstance(output_buffer, list):
                            print(f"- List length: {len(output_buffer)}")
                            print("- Content:")
                            for idx, item in enumerate(output_buffer):
                                if isinstance(item, np.ndarray):
                                    print(f"  Item {idx}: shape={item.shape}, dtype={item.dtype}")
                                    #print(f"  Values: {item}")
                                else:
                                    print(f"  Item {idx}: type={type(item)}")
                                    #print(f"  Values: {item}")
                        
                        # Convert list to numpy array if needed
                        if isinstance(output_buffer, list):
                            try:
                                output_buffer = np.array(output_buffer)
                            except Exception as e:
                                print(f"Could not convert to numpy array: {e}")
                        
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
        # Find apple class index
        try:
            self.apple_class = self.labels.index('apple')
            print(f"Apple class index: {self.apple_class}")
        except ValueError:
            self.apple_class = None
            print("Warning: 'apple' not found in labels")
        self.confidence_threshold = 0.60

    def get_labels(self, labels_path: str) -> list:
        try:
            with open(labels_path, 'r', encoding="utf-8") as f:
                class_names = f.read().splitlines()
            print(f"Loaded {len(class_names)} classes from {labels_path}")
            return class_names
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            return []

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """
        Preprocess image for YOLOv8 inference.
        Expects and maintains 640x640x3 uint8 format.
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure exact size
        if image.shape[:2] != (model_h, model_w):
            image = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure contiguous uint8 array
        image = np.ascontiguousarray(image, dtype=np.uint8)
        
        return image

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        try:
            print("im here!!!!:")
            output_name = list(input_data.keys())[0]
            output_list = input_data.get(output_name)
            print("1212 here!!!!:", output_list)
            print("PPP here!!!!:", output_name)


            
            if (
                output_list is None 
                or not isinstance(output_list, list) 
                or all(
                    isinstance(arr, np.ndarray) and arr.size == 0 
                    if isinstance(arr, np.ndarray) 
                    else True  # Treat non-array items as "empty"
                    for arr in output_list
                )
            ):
                return self._empty_detection_result()
            print("999 here!!!!:")


            
            # Access the nested list containing class detections
            detections_per_class = output_list[0]

            if not isinstance(detections_per_class, list):
                return self._empty_detection_result()

            print("\nProcessing detections:")
            print(f"Number of detection classes: {len(detections_per_class)}")

            boxes = []
            scores = []
            classes = []

            # Loop over each class's detections
            for class_id, detection_array in enumerate(detections_per_class):
                if isinstance(detection_array, np.ndarray) and detection_array.size > 0:
                    print(f"Detections for class ID {class_id} ({self.labels[class_id] if class_id < len(self.labels) else 'Unknown'}):")
                    for detection in detection_array:
                        # Assuming format: [x1, y1, x2, y2, confidence]
                        if len(detection) >= 5:
                            x1, y1, x2, y2, confidence = detection[:5]
                            
                            if confidence > self.confidence_threshold:
                                # Normalize coordinates if they aren't already
                                img_h, img_w = orig_image_shape
                                if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                                    x1, x2 = x1 / img_w, x2 / img_w
                                    y1, y2 = y1 / img_h, y2 / img_h
                                
                                boxes.append([y1, x1, y2, x2])
                                scores.append(float(confidence))
                                classes.append(class_id)
                                
                                print(f"  Found detection: class={class_id}, label={self.labels[class_id]}, conf={confidence:.3f}, "
                                    f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

            result = {
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': len(scores)
            }
            
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

    def format_detections_for_frontend(self, detections: dict, image_shape: tuple) -> List[dict]:
        """Format detections for frontend display."""
        try:
            formatted = []
            
            for i in range(detections['num_detections']):
                ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
                confidence = detections['detection_scores'][i]
                class_id = detections['detection_classes'][i]
                
                # Convert normalized coordinates to pixel values
                img_h, img_w = image_shape[:2]
                ymin_pix, xmin_pix = int(ymin * img_h), int(xmin * img_w)
                ymax_pix, xmax_pix = int(ymax * img_h), int(xmax * img_w)
                
                formatted_detection = {
                    'box': [ymin_pix, xmin_pix, ymax_pix, xmax_pix],
                    'class': self.labels[class_id] if class_id < len(self.labels) else str(class_id),
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
            input_type='UINT8',
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
        try:
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(f"Decoded frame shape: {frame.shape}, dtype: {frame.dtype}")
            return frame
        except Exception as e:
            self.logger.error(f"Frame decoding error: {e}")
            return None

    def process_frame(self, frame_data: str):
        try:
            frame = self.decode_frame(frame_data)
            if frame is None:
                return

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
                            self.logger.info(f"Sending {len(formatted_detections)} detections")
                            self.sio.emit('aiDetections', formatted_detections, namespace='/ai')

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

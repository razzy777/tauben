import argparse
import os
import sys
from pathlib import Path
import numpy as np
import queue
import threading
from PIL import Image
import socketio
import base64
import cv2
import time
import logging
from typing import List, Generator, Optional, Tuple, Dict
from functools import partial
from hailo_platform import (
    HEF, VDevice, FormatType, HailoSchedulingAlgorithm
)

def debug_frame_processing(frame_data: str, save_path: str = 'debug_frames'):
    """Debug helper function to analyze each step of frame processing"""
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 1. Decode base64 frame
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            print(f"Decoded frame: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min()}, {frame.max()}]")
            cv2.imwrite(f'{save_path}/1_decoded_frame.jpg', frame)
            return frame
        else:
            print("Error: Frame decoding failed")
            return None

    except Exception as e:
        print(f"Debug error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        
        # Print model info
        input_shape = self.get_input_shape()
        print(f"Initialized HailoAsyncInference:")
        print(f"- Input shape: {input_shape}")
        print(f"- HEF path: {hef_path}")
        print(f"- Batch size: {batch_size}")

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
                        print(f"Output '{name}' shape: {output_buffer.shape}")
                        print(f"Output '{name}' range: [{output_buffer.min()}, {output_buffer.max()}]")
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
                            break

                        if self.send_original_frame:
                            original_batch, preprocessed_batch = batch_data
                        else:
                            preprocessed_batch = batch_data

                        bindings_list = []
                        for frame in preprocessed_batch:
                            # Ensure frame is C_CONTIGUOUS and float32
                            frame_contiguous = np.ascontiguousarray(frame, dtype=np.float32)
                            print(f"Input frame shape: {frame_contiguous.shape}")
                            print(f"Input frame range: [{frame_contiguous.min()}, {frame_contiguous.max()}]")
                            
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
                    job.wait(10000)
                    
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
                    order='C'
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                    order='C'
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114)):
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        print(f"Initialized ObjectDetectionUtils with {len(self.labels)} labels")

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
            print(f"\nPreprocessing steps for image shape {image.shape}:")
            
            # Save original input
            cv2.imwrite('debug_frames/0_original.jpg', image)
            
            # 1. Resize with aspect ratio maintained
            img_h, img_w = image.shape[:2]
            scale = min(model_w / img_w, model_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"1. Resized to: {resized_image.shape}")
            cv2.imwrite('debug_frames/1_resized.jpg', resized_image)

            # 2. Pad to target size
            delta_w = model_w - new_w
            delta_h = model_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            padded_image = cv2.copyMakeBorder(
                resized_image, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=self.padding_color
            )
            print(f"2. Padded to: {padded_image.shape}")
            cv2.imwrite('debug_frames/2_padded.jpg', padded_image)

            # 3. Convert BGR to RGB
            rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('debug_frames/3_rgb.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            # 4. Normalize to [0, 1]
            normalized_image = rgb_image.astype(np.float32) / 255.0
            print(f"4. Normalized range: [{normalized_image.min()}, {normalized_image.max()}]")

            # 5. Transpose to CHW format
            chw_image = np.transpose(normalized_image, (2, 0, 1))
            print(f"5. Transposed to CHW: {chw_image.shape}")

            # Ensure C-contiguous
            final_image = np.ascontiguousarray(chw_image, dtype=np.float32)
            print(f"Final shape: {final_image.shape}, dtype: {final_image.dtype}")
            print(f"Is C-contiguous: {final_image.flags['C_CONTIGUOUS']}")

            return final_image

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_detections(self, input_data: dict, orig_image_shape: Tuple[int, int]) -> dict:
        """
        Extract detections from model output.
        """
        try:
            boxes = []
            scores = []
            classes = []
            num_detections = 0

            output_tensor = input_data[next(iter(input_data))]
            print(f"\nProcessing output tensor:")
            print(f"- Shape: {output_tensor.shape}")
            print(f"- Type: {output_tensor.dtype}")
            print(f"- Range: [{output_tensor.min()}, {output_tensor.max()}]")

            if output_tensor.size > 0:
                print(f"- First detection raw data: {output_tensor[0]}")

            # Lower confidence threshold for debugging
            confidence_threshold = 0.1

            # Process each detection
            for det in output_tensor:
                x1, y1, x2, y2, confidence, class_id = det[:6]
                score = float(confidence)
                
                if score >= confidence_threshold:
                    print(f"\nValid detection found:")
                    print(f"- Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                    print(f"- Confidence: {score:.2f}")
                    print(f"- Class ID: {int(class_id)}")
                    
                    # Scale boxes to original image size
                    h, w = orig_image_shape
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)

                    boxes.append([y1, x1, y2, x2])
                    scores.append(score)
                    classes.append(int(class_id))
                    num_detections += 1

            result = {
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': num_detections
            }
            
            print(f"\nExtracted {num_detections} detections")
            if num_detections > 0:
                print("First detection details:")
                print(f"- Box: {boxes[0]}")
                print(f"- Class: {self.labels[classes[0]]}")
                print(f"- Score: {scores[0]:.2f}")
            
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
        """Format detections for frontend display."""
        formatted = []
        h, w = image_shape[:2]
        
        for i in range(detections['num_detections']):
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            formatted.append({
                'box': [
                    float(ymin) / h,
                    float(xmin) / w,
                    float(ymax) / h,
                    float(xmax) / w
                ],
                'class': self.labels[detections['detection_classes'][i]],
                'score': float(detections['detection_scores'][i])
            })
            
        return formatted

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
        
        # Create debug directory
        os.makedirs('debug_frames', exist_ok=True)

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

    def process_frame(self, frame_data: str):
        try:
            # 1. Decode and debug frame
            frame = debug_frame_processing(frame_data)
            if frame is None:
                return

            # 2. Preprocess frame
            try:
                preprocessed_frame = self.utils.preprocess(frame, self.width, self.height)
                print("\nPreprocessed frame stats:")
                print(f"- Shape: {preprocessed_frame.shape}")
                print(f"- Range: [{preprocessed_frame.min():.3f}, {preprocessed_frame.max():.3f}]")
                print(f"- Mean: {preprocessed_frame.mean():.3f}")
            except Exception as e:
                self.logger.error(f"Preprocessing error: {e}")
                return

            # 3. Queue frame for inference
            try:
                self.input_queue.put(([frame], [preprocessed_frame]), block=False)
                print("\nFrame queued for inference")
            except queue.Full:
                self.logger.warning("Input queue full, skipping frame")
                return

            # 4. Get inference results
            try:
                original_frame, outputs = self.output_queue.get(timeout=2.0)
                print("\nReceived inference results:")
                for name, tensor in outputs.items():
                    print(f"Output '{name}':")
                    print(f"- Shape: {tensor.shape}")
                    print(f"- Type: {tensor.dtype}")
                    print(f"- Range: [{tensor.min()}, {tensor.max()}]")
                    if tensor.size > 0:
                        print(f"- First row: {tensor[0]}")

                # 5. Process detections
                if outputs:
                    detections = self.utils.extract_detections(outputs, frame.shape[:2])
                    
                    if detections['num_detections'] > 0:
                        formatted_detections = self.utils.format_detections_for_frontend(
                            detections, frame.shape
                        )
                        self.sio.emit('aiDetections', formatted_detections, namespace='/ai')
                        print(f"\nEmitted {len(formatted_detections)} detections to frontend")
                    else:
                        print("\nNo detections found")
                else:
                    print("\nNo outputs from model")

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

    # Create output directory if needed
    os.makedirs("debug_frames", exist_ok=True)

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
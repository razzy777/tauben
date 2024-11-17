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
from typing import Optional, Dict, Tuple
from functools import partial
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

class HailoDetector:
    def __init__(self, hef_path: str, labels_path: str, server_url: str = 'http://localhost:3000'):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('HailoDetector')
        
        # Initialize queues
        self.input_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.output_queue = queue.Queue(maxsize=10)
        
        # Initialize components
        self.utils = ObjectDetectionUtils(labels_path)
        self.setup_hailo(hef_path)
        self.setup_socketio(server_url)
        
        # Get model dimensions
        self.height, self.width, _ = self.hailo_inference.get_input_shape()
        
        # Threading control
        self.running = False
        self.inference_thread = None

    def setup_hailo(self, hef_path: str):
        try:
            self.logger.info("Initializing Hailo inference...")
            self.hailo_inference = HailoAsyncInference(
                hef_path=hef_path,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                batch_size=1,
                input_type='FLOAT32',
                send_original_frame=True
            )
            self.logger.info("Hailo inference initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def setup_socketio(self, server_url: str):
        self.sio = socketio.Client()
        self.server_url = server_url
        self.connected = False

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
            # Decode frame
            frame = self.decode_frame(frame_data)
            if frame is None:
                return

            # Preprocess frame
            preprocessed_frame = self.utils.preprocess(frame, self.width, self.height)

            # Check if input queue is full
            try:
                self.input_queue.put(([frame], [preprocessed_frame]), block=False)
            except queue.Full:
                self.logger.warning("Input queue full, skipping frame")
                return

            # Get results (with timeout)
            try:
                original_frame, outputs = self.output_queue.get(timeout=1.0)
                self.handle_detection_results(original_frame[0], outputs)
            except queue.Empty:
                self.logger.warning("Timeout waiting for detection results")

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        try:
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode frame")
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Error decoding frame: {e}")
            return None

    def handle_detection_results(self, frame: np.ndarray, outputs: Dict):
        try:
            # Extract detections
            detections = self.utils.extract_detections(outputs, frame.shape[:2])

            if detections['num_detections'] > 0:
                # Format detections for frontend
                formatted_detections = []
                for i in range(detections['num_detections']):
                    detection = {
                        'box': [
                            detections['detection_boxes'][i][1] / frame.shape[1],  # xmin normalized
                            detections['detection_boxes'][i][0] / frame.shape[0],  # ymin normalized
                            detections['detection_boxes'][i][3] / frame.shape[1],  # xmax normalized
                            detections['detection_boxes'][i][2] / frame.shape[0]   # ymax normalized
                        ],
                        'class': self.utils.labels[detections['detection_classes'][i]],
                        'score': float(detections['detection_scores'][i])
                    }
                    formatted_detections.append(detection)

                # Send to frontend
                if self.connected:
                    self.sio.emit('aiDetections', formatted_detections, namespace='/ai')
                    self.logger.debug(f"Sent {len(formatted_detections)} detections")

        except Exception as e:
            self.logger.error(f"Error handling detection results: {e}")

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

def main():
    parser = argparse.ArgumentParser(description="Hailo YOLOv8 Object Detection")
    parser.add_argument("--net", help="Path to HEF file", default="/home/johannes/Downloads/yolov8n.hef")
    parser.add_argument("--labels", help="Path to labels file", default="coco.txt")
    parser.add_argument("--server", help="Server URL", default="http://localhost:3000")
    args = parser.parse_args()

    detector = HailoDetector(
        hef_path=args.net,
        labels_path=args.labels,
        server_url=args.server
    )
    
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()

if __name__ == "__main__":
    main()
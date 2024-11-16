import cv2
import numpy as np
import socketio
import base64
import queue
from typing import Dict, Optional
from hailo_platform import (
    HEF,
    VDevice,
    FormatType,
    HailoSchedulingAlgorithm
)
from functools import partial

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
        
        # Create VDevice with round-robin scheduling
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        # Initialize HEF and create inference model
        print(f"Loading HEF from: {hef_path}")
        self.hef = HEF(hef_path)
        
        # Print network information before creating device
        input_vstream_infos = self.hef.get_input_vstream_infos()
        output_vstream_infos = self.hef.get_output_vstream_infos()
        
        print(f"Model input shape: {input_vstream_infos[0].shape}")
        print(f"Model input format: {input_vstream_infos[0].format}")
        
        # Create VDevice
        self.target = VDevice(params)
        print("VDevice created successfully")
        
        # Create inference model
        self.infer_model = self.target.create_infer_model(hef_path)
        print("Created inference model")
        
        # Set batch size
        self.infer_model.set_batch_size(1)
        print(f"Set batch size to 1")
        
        # Set input/output types if specified
        if input_type is not None:
            self._set_input_type(input_type)
            print(f"Set input type to: {input_type}")
            
        if output_type is not None:
            self._set_output_type(output_type)
            print(f"Set output types: {output_type}")
            
        self.output_type = output_type
        self.send_original_frame = send_original_frame
        
        # Print stream information
        print("\nInput Streams:")
        for i, info in enumerate(input_vstream_infos):
            print(f"- Stream {i}: shape={info.shape}, format={info.format}")
        
        print("\nOutput Streams:")
        for i, info in enumerate(output_vstream_infos):
            print(f"- Stream {i}: shape={info.shape}, format={info.format}")

    def _set_input_type(self, input_type: str) -> None:
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Dict[str, str]) -> None:
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def get_input_shape(self):
        return self.hef.get_input_vstream_infos()[0].shape

    def callback(self, completion_info, bindings_list: list, input_batch: list) -> None:
        if completion_info.exception:
            print(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                try:
                    if len(bindings._output_names) == 1:
                        result = bindings.output(bindings._output_names[0]).get_buffer()
                    else:
                        result = {
                            name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                            for name in bindings._output_names
                        }
                    self.output_queue.put((input_batch[i], result))
                except Exception as e:
                    print(f"Error in callback processing result {i}: {e}")

    def _create_bindings(self, configured_infer_model):
        try:
            output_vstream_infos = self.hef.get_output_vstream_infos()
            if self.output_type is None:
                output_buffers = {
                    output_vstream_info.name: np.empty(
                        tuple(self.infer_model.output(output_vstream_info.name).shape),
                        dtype=np.dtype('float32')
                    )
                    for output_vstream_info in output_vstream_infos
                }
            else:
                output_buffers = {
                    name: np.empty(
                        tuple(self.infer_model.output(name).shape),
                        dtype=np.dtype(self.output_type[name].lower())
                    )
                    for name in self.output_type
                }
            return configured_infer_model.create_bindings(output_buffers=output_buffers)
        except Exception as e:
            print(f"Error creating bindings: {e}")
            raise

    def run(self) -> None:
        try:
            with self.infer_model.configure() as configured_infer_model:
                print("Model configured successfully")
                while True:
                    try:
                        batch_data = self.input_queue.get()
                        if batch_data is None:
                            break  # Stop signal
                        
                        if self.send_original_frame:
                            original_batch, preprocessed_batch = batch_data
                        else:
                            preprocessed_batch = batch_data

                        # Process one frame at a time
                        for i, frame in enumerate(preprocessed_batch):
                            try:
                                bindings = self._create_bindings(configured_infer_model)
                                bindings.input().set_buffer(np.array(frame))
                                
                                # Run inference on single frame
                                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                                job = configured_infer_model.run_async(
                                    [bindings],
                                    partial(
                                        self.callback,
                                        input_batch=[original_batch[i]] if self.send_original_frame else [preprocessed_batch[i]],
                                        bindings_list=[bindings]
                                    )
                                )
                                job.wait(10000)
                            except Exception as e:
                                print(f"Error processing frame {i}: {e}")
                                continue
                            
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in inference thread: {e}")
            import traceback
            traceback.print_exc()

def preprocess_frame(frame: np.ndarray, target_shape) -> np.ndarray:
    """Preprocess frame for YOLOv5 inference."""
    target_height, target_width = target_shape[0:2]
    
    # Ensure frame is contiguous and in the right format
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Resize while maintaining aspect ratio
    height, width = frame.shape[:2]
    scale = min(target_width/width, target_height/height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create black image with target size
    new_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image in center
    new_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    print(f"Preprocessed image shape: {new_img.shape}, dtype: {new_img.dtype}")
    return new_img

def init_hailo():
    try:
        print("Starting Hailo initialization...")
        # Create queues for async inference
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Initialize hailo inference
        hailo_inference = HailoAsyncInference(
            hef_path='/home/johannes/Downloads/yolov5s.hef',
            input_queue=input_queue,
            output_queue=output_queue,
            batch_size=1,
            input_type='UINT8',  # YOLOv5 typically expects UINT8 input
            send_original_frame=True
        )
        
        return hailo_inference, input_queue, output_queue

    except Exception as e:
        print(f"Failed to initialize Hailo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

sio = socketio.Client(
    logger=True,
    engineio_logger=True,
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1
)

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

        # Get input shape
        input_shape = hailo_inference.get_input_shape()
        
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame, input_shape)
        
        # Put frame in input queue
        input_queue.put(([frame], [preprocessed_frame]))
        
        # Get results from output queue
        original_frame, outputs = output_queue.get()
        
        if outputs:
            # Convert outputs to detections format expected by the server
            detections = process_detections(outputs, frame.shape)
            print("Processing detection results...", detections)

            if detections:
                # Emit detections to server
                sio.emit('aiDetections', detections, namespace='/ai')
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

def process_detections(outputs, frame_shape):
    """
    Process YOLOv5 outputs into a format expected by the server.
    Returns a list of detections with normalized coordinates.
    """
    height, width = frame_shape[:2]
    detections = []
    
    # If outputs is a dictionary (multiple output tensors)
    if isinstance(outputs, dict):
        # Process YOLOv5 output format
        # This needs to be adjusted based on your model's specific output format
        for output_tensor in outputs.values():
            # Each detection should be: [x1, y1, x2, y2, conf, class_id]
            for detection in output_tensor:
                print(f"Detection entry: {detection}")
                if len(detection) >= 6:  # Make sure we have enough elements
                    confidence = float(detection[4])
                    if confidence > 0.1:  # Confidence threshold
                        # Normalize coordinates to 0-1 range
                        x1 = float(detection[0]) / width
                        y1 = float(detection[1]) / height
                        x2 = float(detection[2]) / width
                        y2 = float(detection[3]) / height
                        class_id = int(detection[5])
                        
                        detections.append({
                            'box': [y1, x1, y2, x2],  # Format expected by server
                            'confidence': confidence,
                            'class_id': class_id
                        })
    
    return detections

def main():
    try:
        # Initialize Hailo
        global hailo_inference, input_queue, output_queue
        hailo_inference, input_queue, output_queue = init_hailo()

        if not hailo_inference:
            print("Failed to initialize Hailo device. Exiting...")
            return

        # Connect to the Node.js server's AI namespace
        print("\nConnecting to server...")
        connection_url = 'http://localhost:3000'
        print(f"Attempting to connect to {connection_url}")
        
        sio.connect(
            connection_url,
            namespaces=['/ai'],
            wait_timeout=10,
            transports=['websocket', 'polling'],
            socketio_path='/socket.io',
            headers={
                'Content-Type': 'application/json',
            }
        )
        print("Connected successfully")
        
        # Start inference thread
        import threading
        inference_thread = threading.Thread(target=hailo_inference.run)
        inference_thread.start()
        print("Started inference thread")
        
        # Keep the connection alive
        print("Waiting for events...")
        sio.wait()

    except socketio.exceptions.ConnectionError as e:
        print(f"Socket.IO Connection error: {e}")
        print("Error details:", getattr(e, 'args', ['No details available']))
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if input_queue:
            input_queue.put(None)  # Signal inference thread to stop
        if 'inference_thread' in locals():
            inference_thread.join()
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main()

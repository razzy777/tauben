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
        
        # Create VDevice with specific parameters
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # Set buffer size constraints
        params.max_desc_page_size = 4096  # Match hardware limit
        
        # Initialize HEF and create inference model
        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        
        # Set input/output types if specified
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)
            
        self.output_type = output_type
        self.send_original_frame = send_original_frame
        
        # Print model information
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        print(f"Model input shape: {input_vstream_info.shape}")
        print(f"Model input format: {input_vstream_info.format}")

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
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                self.output_queue.put((input_batch[i], result))

    def _create_bindings(self, configured_infer_model):
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(np.dtype(output_info.format.type.name.lower()))
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=np.dtype(self.output_type[name].lower())
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

    def run(self) -> None:
        try:
            with self.infer_model.configure() as configured_infer_model:
                while True:
                    batch_data = self.input_queue.get()
                    if batch_data is None:
                        break  # Stop signal
                    
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
                        bindings_list,
                        partial(
                            self.callback,
                            input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                            bindings_list=bindings_list
                        )
                    )
                job.wait(10000)  # Wait for the last job
        except Exception as e:
            print(f"Error in inference thread: {e}")
            import traceback
            traceback.print_exc()

def preprocess_frame(frame: np.ndarray, target_shape) -> np.ndarray:
    """Preprocess frame for YOLOv5 inference."""
    # Ensure we're using uint8 data type
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
        
    # Resize while maintaining aspect ratio
    input_height, input_width = target_shape[1:3]
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(input_width/width, input_height/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create empty image with target size
    new_img = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (input_height - new_height) // 2
    x_offset = (input_width - new_width) // 2
    
    # Place resized image in center
    new_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
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
            send_original_frame=True  # We want to keep the original frame for visualization
        )
        
        return hailo_inference, input_queue, output_queue

    except Exception as e:
        print(f"Failed to initialize Hailo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Initialize Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('videoFrame')
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
        # Send both original and preprocessed frames since send_original_frame=True
        input_queue.put(([frame], [preprocessed_frame]))
        
        # Get results from output queue
        original_frame, outputs = output_queue.get()
        
        if outputs:
            print(f"Got inference results")
            # TODO: Process detections here
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

def main():
    try:
        # Initialize Hailo
        global hailo_inference, input_queue, output_queue
        hailo_inference, input_queue, output_queue = init_hailo()

        if not hailo_inference:
            print("Failed to initialize Hailo device. Exiting...")
            return

        # Connect to the Node.js server
        print("\nConnecting to server...")
        sio.connect('http://localhost:3000')
        print("Connected successfully")
        
        # Start inference thread
        import threading
        from functools import partial
        inference_thread = threading.Thread(target=hailo_inference.run)
        inference_thread.start()
        
        # Keep the connection alive
        sio.wait()

    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if input_queue:
            input_queue.put(None)  # Signal inference thread to stop
        if 'inference_thread' in locals():
            inference_thread.join()
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main()
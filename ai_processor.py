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
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        print(f"Model input shape: {input_vstream_info.shape}")
        print(f"Model input format: {input_vstream_info.format}")
        
        # Create VDevice
        self.target = VDevice(params)
        print("VDevice created successfully")
        
        # Create inference model with smaller batch size
        self.infer_model = self.target.create_infer_model(hef_path)
        print("Created inference model")
        
        # Set very small batch size to reduce memory requirements
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
        for name, info in self.hef.get_input_vstream_infos().items():
            print(f"- {name}: shape={info.shape}, format={info.format}")
        
        print("\nOutput Streams:")
        for name, info in self.hef.get_output_vstream_infos().items():
            print(f"- {name}: shape={info.shape}, format={info.format}")

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
                        result = bindings.output().get_buffer()
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
            if self.output_type is None:
                output_buffers = {
                    output_info.name: np.empty(
                        tuple(self.infer_model.output(output_info.name).shape),
                        dtype=np.dtype('float32')  # Default to float32
                    )
                    for output_info in self.hef.get_output_vstream_infos()
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
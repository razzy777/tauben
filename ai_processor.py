import cv2
import numpy as np
import socketio
import base64
from hailo_platform.pyhailort import (
    Device,
    VDevice,
    ConfigureParams,
    InferVStreams,
    HailoRTException,
    HEF
)

def init_hailo():
    try:
        print("Initializing Hailo device...")

        # Create Device
        device = Device()
        device_id = device.device_id
        print(f"Found device with ID: {device_id}")

        # Create VDevice
        vdevice = VDevice(device)  # Pass device to VDevice constructor
        print("VDevice created successfully")

        # Load YOLOv5 HEF file
        hef_path = '/home/johannes/Downloads/yolov5s.hef'
        print(f"Loading HEF file from: {hef_path}")
        hef = HEF(hef_path)
        print("HEF loaded successfully")

        # Get network group names
        network_group_names = hef.get_network_group_names()
        print(f"Network groups found: {network_group_names}")

        # Configure params
        configure_params = ConfigureParams()
        configure_params.stream_interface = device.get_default_stream_interface()
        configure_params.batch_size = 1

        # Configure the device
        network_groups = vdevice.configure(hef, configure_params)
        network_group = network_groups[0]  # Get first network group
        print("Network configured successfully")

        # Print available stream information
        input_vstreams = network_group.input_vstream_infos
        output_vstreams = network_group.output_vstream_infos
        
        print("\nInput Streams:")
        for name, info in input_vstreams.items():
            print(f"- {name}: shape={info.shape}, format={info.format}")
            
        print("\nOutput Streams:")
        for name, info in output_vstreams.items():
            print(f"- {name}: shape={info.shape}, format={info.format}")

        return vdevice, network_group

    except HailoRTException as e:
        print(f"Failed to initialize Hailo device: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during Hailo initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Initialize Socket.IO client
sio = socketio.Client()

# Socket.IO event handlers
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

        # Get input shape from network
        input_info = next(iter(network_group.input_vstream_infos.values()))
        input_shape = tuple(input_info.shape[1:3])  # Height, Width
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame, input_shape)

        # Run inference
        if network_group:
            outputs = run_inference(network_group, processed_frame)
            if outputs:
                detections = postprocess(outputs, input_shape)
                if detections:
                    print(f"Found {len(detections)} detections")
                    sio.emit('aiDetections', detections)
        else:
            print("Hailo network not configured")

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

def main():
    try:
        # Initialize Hailo
        print("Starting Hailo initialization...")
        global device, network_group
        device, network_group = init_hailo()

        if not device or not network_group:
            print("Failed to initialize Hailo device. Exiting...")
            return

        # Connect to the Node.js server
        print("\nConnecting to server...")
        sio.connect('http://localhost:3000')
        print("Connected successfully")
        
        # Keep the connection alive
        sio.wait()

    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main()
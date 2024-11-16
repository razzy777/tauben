import cv2
import numpy as np
import socketio
import base64
from hailo_platform.pyhailort.pyhailort import VDevice, HailoRTException, Device

# Initialize Socket.IO client
sio = socketio.Client()

def init_hailo():
    try:
        print("Initializing Hailo device...")

        # Create Device first
        device = Device()
        print("Device created successfully")

        # Create VDevice from Device
        vdevice = VDevice(device)
        print("VDevice created successfully")

        # Path to your HEF file
        hef_path = '/home/johannes/tauben/venv/lib/python3.9/site-packages/hailo_tutorials/hefs/resnet_v1_18.hef'
        print(f"Loading HEF file from: {hef_path}")

        # Configure the device with the HEF file
        network_groups = vdevice.configure_from_file(hef_path)
        network_group = network_groups[0]  # Get first network group
        print("Network configured")

        # Get input and output information
        input_vstreams_info = network_group.input_vstream_infos
        output_vstreams_info = network_group.output_vstream_infos
        print("Input VStreams:", input_vstreams_info)
        print("Output VStreams:", output_vstreams_info)

        return vdevice, network_group
    except HailoRTException as e:
        print(f"Failed to initialize Hailo device: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during Hailo initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Initialize Hailo device
print("Starting Hailo initialization...")
device, network_group = init_hailo()

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

        # Preprocess frame
        processed_frame = preprocess_frame(frame)

        # Run inference
        if network_group:
            detections = run_inference(processed_frame)
            if detections:
                print(f"Found {len(detections)} detections")
                sio.emit('aiDetections', detections)
        else:
            print("Hailo network not configured")

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

def preprocess_frame(frame):
    """Preprocess frame for ResNet inference"""
    # Resize to model input size
    resized = cv2.resize(frame, (224, 224))
    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    return batched

def run_inference(preprocessed_frame):
    """Run inference on Hailo device"""
    try:
        # Get input name from input vstream info
        input_name = list(network_group.input_vstream_infos.keys())[0]
        
        # Create input data dictionary
        input_data = {input_name: preprocessed_frame}
        
        # Create VStreams context and run inference
        with network_group.create_vstreams() as vstreams:
            outputs = vstreams.infer(input_data)
            print(f"Inference outputs: {outputs.keys()}")
        
        # Process outputs
        detections = process_outputs(outputs)
        return detections

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_outputs(outputs):
    """Process ResNet outputs"""
    detections = []
    
    try:
        # Get output tensor name
        output_name = list(outputs.keys())[0]
        scores = outputs[output_name][0]  # Get scores from first batch
        
        # Get top prediction
        max_score_idx = np.argmax(scores)
        max_score = float(scores[max_score_idx])
        
        if max_score > 0.5:  # Confidence threshold
            detection = {
                'class': int(max_score_idx),
                'score': max_score
            }
            detections.append(detection)
            print(f"Detected class {max_score_idx} with confidence {max_score}")

    except Exception as e:
        print(f"Error processing outputs: {e}")
        import traceback
        traceback.print_exc()
    
    return detections

def main():
    try:
        if not device or not network_group:
            print("Failed to initialize Hailo device. Exiting...")
            return

        # Print device information
        print("\nDevice Information:")
        print("-------------------")
        print(f"Input VStreams: {network_group.input_vstream_infos}")
        print(f"Output VStreams: {network_group.output_vstream_infos}")

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
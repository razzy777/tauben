import cv2
import numpy as np
import socketio
import base64
from hailo_platform.pyhailort.pyhailort import Device, HailoRTException

# Initialize Socket.IO client
sio = socketio.Client()

def init_hailo():
    try:
        print("Initializing Hailo device...")

        # Create Device (not VDevice)
        device = Device()
        print("Device created successfully")

        # Path to your HEF file - using one from the installed package
        hef_path = '/home/johannes/tauben/venv/lib/python3.9/site-packages/hailo_tutorials/hefs/resnet_v1_18.hef'
        print(f"Loading HEF file from: {hef_path}")

        # Configure the device with the HEF
        network_group = device.configure(hef_path)
        print("Network configured")

        return device, network_group
    except HailoRTException as e:
        print(f"Failed to initialize Hailo device: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during Hailo initialization: {e}")
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

def preprocess_frame(frame):
    """Preprocess frame for Hailo inference"""
    # Resize to model input size (adjust as needed)
    resized = cv2.resize(frame, (224, 224))  # ResNet input size
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
        # Create input tensors - adjust name based on model
        input_data = {'input': preprocessed_frame}
        
        # Run inference
        outputs = network_group.infer(input_data)
        print(f"Inference outputs: {outputs.keys()}")  # Debug print
        
        # Process outputs
        detections = process_outputs(outputs)
        return detections

    except Exception as e:
        print(f"Inference error: {e}")
        return []

def process_outputs(outputs):
    """Process model outputs into detections"""
    detections = []
    
    try:
        # For ResNet, output is typically classification scores
        scores = outputs.get('output', [])
        
        # Get top prediction
        if len(scores) > 0:
            max_score_idx = np.argmax(scores)
            max_score = float(scores[max_score_idx])
            
            if max_score > 0.5:  # Confidence threshold
                detection = {
                    'class': int(max_score_idx),
                    'score': max_score
                }
                detections.append(detection)

    except Exception as e:
        print(f"Error processing outputs: {e}")
    
    return detections

def main():
    try:
        if not device or not network_group:
            print("Failed to initialize Hailo device. Exiting...")
            return

        # Connect to the Node.js server
        print("Connecting to server...")
        sio.connect('http://localhost:3000')
        print("Connected successfully")
        
        # Keep the connection alive
        sio.wait()

    except Exception as e:
        print(f"Main error: {e}")
    finally:
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main()
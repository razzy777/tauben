import cv2
import numpy as np
import socketio
import base64
import hailo_platform.pyhailort as pyhailort


# Initialize Socket.IO client
sio = socketio.Client()

def init_hailo():
    try:
        print("Initializing Hailo device...")
        # Initialize Hailo device
        device = pyhailort.Device()
        
        # Path to your HEF file
        hef_path = '/home/johannes/hailo_models/yolov5_person.hef'
        
        # Configure the network
        network_group = device.configure(hef_path)
        network_group_params = network_group.create_params()
        
        print("Hailo device initialized successfully")
        return device, network_group
    except Exception as e:
        print(f"Failed to initialize Hailo device: {e}")
        return None, None

# Initialize Hailo device
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
    resized = cv2.resize(frame, (640, 640))
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
        # Prepare input
        input_data = {'input': preprocessed_frame}
        
        # Run inference
        outputs = network_group.infer(input_data)
        
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
        # Get outputs (adjust keys based on your model)
        boxes = outputs.get('boxes', [])
        scores = outputs.get('scores', [])
        classes = outputs.get('classes', [])

        # Filter detections
        confidence_threshold = 0.5
        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                detection = {
                    'box': boxes[i].tolist(),
                    'score': float(scores[i]),
                    'class': int(classes[i])
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
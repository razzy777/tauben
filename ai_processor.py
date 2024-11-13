import cv2
import numpy as np
import hailo_platform
import socketio

# Initialize Socket.IO client
sio = socketio.Client()

# Load the Hailo model
hef_path = '/home/johannes/hailo_models/yolov5_person.hef'  # Replace with the path to your HEF file
print("Loading Hailo HEF model...")
device = hailo_platform.Device()
network_group = device.configure(hef_path)
network_group_params = network_group.create_params()
print("Model loaded successfully")

# Event handler for connecting to the /ai namespace
@sio.event(namespace='/ai')
def connect():
    print('Connected to AI namespace on Node.js server')

# Event handler for disconnecting from the /ai namespace
@sio.event(namespace='/ai')
def disconnect():
    print('Disconnected from AI namespace on Node.js server')

# Event handler for connection errors
@sio.event
def connect_error(data):
    print("Connection failed:", data)

# Event handler for receiving video frames
@sio.on('videoFrame', namespace='/ai')
def on_video_frame(data):
    print("Received frame from Node.js")

    # Convert the received frame data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error: Received empty frame")
        return

    # Preprocess frame
    img_resized = cv2.resize(frame, (640, 640))  # Resize to model's input resolution
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0  # Normalize if required by the model
    img_expanded = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    # Run inference using Hailo SDK
    input_tensor = {'input': img_expanded}
    output = network_group.infer(input_tensor)

    # Process detections (assuming YOLO output format)
    boxes = output['boxes']
    classes = output['class_ids']
    scores = output['scores']

    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 0:  # Assuming class 0 is 'person'
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({
                'class': 'person',
                'score': float(scores[i]),
                'box': [float(ymin), float(xmin), float(ymax), float(xmax)],
            })

    # Emit detections to the Node.js server
    if detections:
        print("Emitting detections:", detections)
        sio.emit('aiDetections', detections, namespace='/ai')

def main():
    try:
        # Connect to the Node.js server, specifying the namespace
        sio.connect('http://localhost:3000', namespaces=['/ai'])
        sio.wait()  # Keep the script running to listen for events

    except Exception as e:
        print("An error occurred:", e)
    finally:
        sio.disconnect()

if __name__ == '__main__':
    main()

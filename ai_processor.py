import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socketio

# Initialize Socket.IO client
sio = socketio.Client()

# Load the TFLite model globally
interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully")

@sio.event
def connect():
    print('Connected to Node.js server')

@sio.event
def disconnect():
    print('Disconnected from Node.js server')

@sio.event
def connect_error(data):
    print("Connection failed:", data)

# Event handler for receiving video frames
@sio.on('videoFrame')
def on_video_frame(data):
    print("Received frame from Node.js")  # Add logging
    # Convert the received frame data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error: Received empty frame")
        return

    # Preprocess frame
    img_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_expanded = np.expand_dims(img_rgb, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_expanded)

    # Run inference
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Process detections
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
        sio.emit('aiDetections', detections)

def main():
    try:
        # Connect to the Node.js server (use /ai namespace)
        sio.connect('http://localhost:3000/ai')  # Ensure the namespace is correct
        sio.wait()  # Keep the script running to listen for events

    except Exception as e:
        print("An error occurred:", e)
    finally:
        sio.disconnect()

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socketio
import eventlet
import time

# Initialize Socket.IO server
sio = socketio.Server()

# Wrap with a WSGI application
app = socketio.WSGIApp(sio)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded and ready")

@sio.event
def connect(sid, environ):
    print('Node.js client connected:', sid)

@sio.event
def disconnect(sid):
    print('Node.js client disconnected:', sid)

@sio.event
def videoFrame(sid, data):
    # data is the frame sent from Node.js
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error: Received empty frame")
        return

    # Preprocess frame
    img_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_expanded = np.expand_dims(img_rgb, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_expanded)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 0:  # Detect 'person' class
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({
                'class': 'person',
                'score': float(scores[i]),
                'box': [float(ymin), float(xmin), float(ymax), float(xmax)]
            })

    if detections:
        print("Sending detections:", detections)
        # Send detections back to Node.js
        sio.emit('aiDetections', detections)

# Run the server
if __name__ == '__main__':
    print("Starting AI processor server...")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)

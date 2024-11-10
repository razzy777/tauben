import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socketio
import time

# Initialize Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to Node.js server')

@sio.event
def disconnect():
    print('Disconnected from Node.js server')

try:
    # Connect to the Node.js server
    sio.connect('http://localhost:3000')  # Adjust the URL if needed

    # Load the TFLite model
    try:
        print("Attempting to load model...")
        interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print("Failed to load model:", e)
        raise SystemExit("Exiting due to model load failure.")

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model input details:", input_details)
    print("Model output details:", output_details)

    # Initialize camera
    print("Attempting to open the camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        raise SystemExit("Exiting due to camera initialization failure.")
    print("Camera initialized successfully")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame could not be read from the camera.")
                break

            # Preprocess frame
            input_shape = input_details[0]['shape']
            img_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_expanded = np.expand_dims(img_rgb, axis=0)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_expanded)

            # Run inference
            interpreter.invoke()

            # Get detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
            classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
            scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

            # Process detections
            detections = []
            for i in range(len(scores)):
                if scores[i] > 0.5 and classes[i] == 0:  # Class 0 is 'person' in COCO dataset
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

            # Sleep briefly to reduce CPU usage
            time.sleep(0.1)

    except Exception as e:
        print("An error occurred during processing:", e)

    finally:
        print("Releasing camera and disconnecting.")
        cap.release()
        sio.disconnect()

except Exception as e:
    print("An error occurred:", e)

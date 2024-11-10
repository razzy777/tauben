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

@sio.event
def connect_error(data):
    print("Connection failed:", data)

def main():
    try:
        # Connect to the Node.js server (use /ai namespace)
        sio.connect('http://localhost:3000/ai')  # Adjust the URL if needed

        # Load the TFLite model
        interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame could not be read from the camera.")
                break

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
                if scores[i] > 0.5 and classes[i] == 0:
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
        print("An error occurred:", e)
    finally:
        cap.release()
        sio.disconnect()

if __name__ == '__main__':
    main()

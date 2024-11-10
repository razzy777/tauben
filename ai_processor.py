import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socketio
import subprocess
import os
import time

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to Node.js server')

@sio.event
def disconnect():
    print('Disconnected from Node.js server')

try:
    sio.connect('http://localhost:3000')  # Adjust if needed
    interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded and ready")

    def capture_frame():
        image_path = '/tmp/frame.jpg'
        capture_command = f'libcamera-still -o {image_path} -t 1 --width 640 --height 480'
        try:
            subprocess.run(capture_command, shell=True, check=True)
            frame = cv2.imread(image_path)
            os.remove(image_path)  # Clean up the image after reading
            return frame
        except subprocess.CalledProcessError as e:
            print("Error capturing frame:", e)
            return None

    while True:
        frame = capture_frame()
        if frame is None:
            print("Error: Frame could not be captured.")
            break

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
            sio.emit('aiDetections', detections)

        time.sleep(0.1)

    sio.disconnect()

except Exception as e:
    print("An error occurred:", e)

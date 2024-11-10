import logging
import socketio
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up connection to Node.js server
sio = socketio.Client()

@sio.event
def connect():
    logger.info("Connected to Node.js server")

@sio.event
def disconnect():
    logger.info("Disconnected from Node.js server")

# Load the model
interpreter = tflite.Interpreter(model_path="/home/johannes/tflite_models/detect.tflite")
interpreter.allocate_tensors()
logger.info("Model loaded successfully")
logger.debug(f"Model input details: {interpreter.get_input_details()}")
logger.debug(f"Model output details: {interpreter.get_output_details()}")

# Frame capture and processing
def capture_frame():
    logger.debug("Attempting to capture a frame...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Failed to capture frame")
        return None

    logger.info("Frame captured successfully")
    return frame

# Inference and detection
def process_frame(frame):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    logger.debug("Converting frame to tensor and resizing...")
    input_tensor = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Collect detections
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]

    logger.debug(f"Raw detection boxes: {boxes}")
    logger.debug(f"Raw detection scores: {scores}")
    logger.debug(f"Raw detection classes: {classes}")

    detections = []
    for i in range(len(scores)):
        score = scores[i]
        if score > 0.5:  # Adjust threshold as necessary
            klass = int(classes[i])
            box = boxes[i]
            detections.append({"class": klass, "score": float(score), "box": [float(b) for b in box]})
            logger.info(f"Detection: class={klass}, score={score}, box={box}")

    if detections:
        sio.emit("aiDetections", detections)
        logger.info(f"Emitting detections: {detections}")
    else:
        logger.info("No detections above threshold")

# Run the loop
if __name__ == '__main__':
    sio.connect('http://localhost:3000')
    while True:
        frame = capture_frame()
        if frame is not None:
            process_frame(frame)
        else:
            logger.error("No frame to process")

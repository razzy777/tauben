import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socketio
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Socket.IO client
sio = socketio.Client()

# Load the TFLite model globally
interpreter = tflite.Interpreter(model_path='/home/johannes/tflite_models/detect.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("Model loaded successfully")
logger.debug(f"Model input details: {interpreter.get_input_details()}")
logger.debug(f"Model output details: {interpreter.get_output_details()}")


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
    # Convert the received frame data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logger.debug("Attempting to capture a frame...")

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
        score = scores[i]
        if score > 0.5:  # Modify threshold as needed
            klass = classes[i]
            box = output_data[i]
            detections.append({
                "class": klass,
                "score": float(score),
                "box": [float(b) for b in box]
            })
            logger.info(f"Detection: class={klass}, score={score}, box={box}")

    # Emit detections if there are any
    if detections:
        sio.emit('aiDetections', detections)
        logger.info(f"Emitting detections: {detections}")
    else:
        logger.info("No detections above threshold")

def main():
    try:
        # Connect to the Node.js server (use /ai namespace)
        sio.connect('http://localhost:3000/ai')  # Adjust the URL if needed
        sio.wait()  # Keep the script running to listen for events

    except Exception as e:
        print("An error occurred:", e)
    finally:
        sio.disconnect()

if __name__ == '__main__':
    main()

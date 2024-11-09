const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const tflite = require('@tensorflow/tfjs');

// Existing folder and path definitions
const folderPath = '/home/johannes/tauben/images';
const imagePath = path.join(folderPath, 'test_picture.jpg');

// Load the model
const modelPath = path.join(__dirname, 'tflite_models', 'detect.tflite');
let detectionModel;

async function loadModel() {
  const modelBuffer = fs.readFileSync(modelPath);
  detectionModel = await tflite.loadTFLiteModel(modelBuffer);
}

loadModel();


// Ensure directory exists with permissions
if (!fs.existsSync(folderPath)) {
    console.log(`Creating directory at: ${folderPath}`);
    fs.mkdirSync(folderPath, { recursive: true });
    fs.chmodSync(folderPath, 0o777);
} else {
    console.log(`Directory exists at: ${folderPath}`);
    fs.chmodSync(folderPath, 0o777);
}

// Keep track of the video process
let videoProcess = null;

// Function to start video stream
function startVideoStream(socket) {
    if (videoProcess) {
        console.log('Video stream already running');
        return;
    }

    console.log('Starting video stream...');
    
    const command = 'libcamera-vid';
    const args = [
        '--codec', 'mjpeg',
        '--width', '1280',
        '--height', '720',
        '--framerate', '15',
        //'--rotation', '180',
        '--inline',
        '--nopreview',
        '--timeout', '0',
        '--output', '-'
    ];

    try {
        videoProcess = spawn(command, args);
        let buffer = Buffer.from([]);

        videoProcess.stdout.on('data', (data) => {
            buffer = Buffer.concat([buffer, data]);
            
            // Look for JPEG start and end markers
            let start = 0;
            let end = 0;
            
            while (true) {
                start = buffer.indexOf(Buffer.from([0xFF, 0xD8]));
                end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));
                
                if (start !== -1 && end !== -1 && end > start) {
                    const frame = buffer.slice(start, end + 2);
                    if (frame.length > 1000) {
                        socket.emit('videoFrame', frame.toString('base64'));
                        console.log('Frame sent, size:', frame.length); // Debug log
                    }
                    buffer = buffer.slice(end + 2);
                } else {
                    break;
                }
            }
        });

		videoProcess.stdout.on('data', async (data) => {
			buffer = Buffer.concat([buffer, data]);
		
			// Process frames
			let start = buffer.indexOf(Buffer.from([0xFF, 0xD8]));
			let end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));
		
			while (start !== -1 && end !== -1 && end > start) {
			  const frame = buffer.slice(start, end + 2);
			  buffer = buffer.slice(end + 2);
		
			  // Emit video frame
			  socket.emit('videoFrame', frame.toString('base64'));
		
			  // Perform object detection
			  const detections = await performObjectDetection(frame);
		
			  // Emit detections
			  if (detections.length > 0) {
				socket.emit('detections', detections);
			  }
		
			  start = buffer.indexOf(Buffer.from([0xFF, 0xD8]));
			  end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));
			}
		  });
		

        videoProcess.on('error', (error) => {
            console.error('Camera process error:', error);
        });

    } catch (error) {
        console.error('Failed to start video stream:', error);
        videoProcess = null;
    }
}

// Function to stop video stream
function stopVideoStream() {
    if (videoProcess) {
        videoProcess.kill('SIGTERM');
        videoProcess = null;
        console.log('Video stream stopped');
    }
}

async function performObjectDetection(imageBuffer) {
	if (!detectionModel) {
	  console.error('Model not loaded yet.');
	  return [];
	}
  
	// Load image into tensor
	const imageTensor = tf.node.decodeImage(imageBuffer, 3);
	const resizedImage = tf.image.resizeBilinear(imageTensor, [300, 300]);
	const normalizedImage = resizedImage.div(255.0).expandDims(0);
  
	// Run inference
	const output = detectionModel.execute(normalizedImage);
  
	// Process output
	const [boxes, classes, scores] = output;
	const detections = [];
  
	for (let i = 0; i < scores.size; i++) {
	  const score = scores.dataSync()[i];
	  if (score > 0.5) {
		const klass = classes.dataSync()[i];
		if (klass === 1) { // 'person' class in COCO dataset
		  const box = boxes.dataSync().slice(i * 4, (i + 1) * 4);
		  detections.push({
			class: 'person',
			score,
			box,
		  });
		}
	  }
	}
  
	// Clean up tensors
	tf.dispose([imageTensor, resizedImage, normalizedImage, output]);
  
	return detections;
  }
  

// Existing photo capture function
async function captureImage() {
    return new Promise((resolve, reject) => {
        const captureCommand = `libcamera-still -o ${imagePath} -t 1000 --width 1280 --height 720`;
        
        exec(captureCommand, (err, stdout, stderr) => {
            if (err) {
                reject(new Error(`Error capturing image: ${err.message}`));
                return;
            }
            if (stderr) {
                console.error('libcamera-still error:', stderr);
            }
            console.log('Image successfully captured and saved at:', imagePath);
            resolve(imagePath);
        });
    });
}

// Existing remove image function
async function removeImage() {
    return new Promise((resolve, reject) => {
        if (fs.existsSync(imagePath)) {
            fs.unlink(imagePath, (err) => {
                if (err) {
                    reject(new Error(`Error removing image: ${err.message}`));
                    return;
                }
                console.log('Image successfully removed:', imagePath);
                resolve();
            });
        } else {
            reject(new Error('Image file does not exist.'));
        }
    });
}

module.exports = {
    captureImage,
    removeImage,
    startVideoStream,
    stopVideoStream
};
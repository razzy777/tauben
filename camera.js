const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const ioClient = require('socket.io-client');

// Connect to Python AI processor
const aiSocket = ioClient('http://localhost:5000'); // Use a different port for Python script

// Handle connection events
aiSocket.on('connect', () => {
  console.log('Connected to AI processor');
});

aiSocket.on('disconnect', () => {
  console.log('Disconnected from AI processor');
});

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
            // Emit video frame to frontend
            socket.emit('videoFrame', frame.toString('base64'));
            // Emit frame to Python AI processor
            aiSocket.emit('videoFrame', frame);
          }
          buffer = buffer.slice(end + 2);
        } else {
          break;
        }
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
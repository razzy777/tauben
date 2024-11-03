const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Define the folder path and image file path
const folderPath = '/home/johannes/tauben/images';
const imagePath = path.join(folderPath, 'test_picture.jpg');

// Ensure the directory has correct permissions
if (!fs.existsSync(folderPath)) {
    console.log(`Creating directory at: ${folderPath}`);
    fs.mkdirSync(folderPath, { recursive: true });
    fs.chmodSync(folderPath, 0o777);
} else {
    console.log(`Directory exists at: ${folderPath}`);
    fs.chmodSync(folderPath, 0o777);
}

// Video stream process holder
let videoProcess = null;

// Function to start video stream
function startVideoStream(socket) {
    if (videoProcess) {
        console.log('Video stream already running');
        return;
    }

    console.log('Starting video stream...');
    
    // Command to start video stream
    // Using libcamera-vid with raw output
    videoProcess = spawn('libcamera-vid', [
        '-t', '0',           // Run indefinitely
        '--width', '640',    // Reduced width for better performance
        '--height', '480',   // Reduced height for better performance
        '--framerate', '15', // Lower framerate for better network performance
        '--codec', 'mjpeg',  // Use MJPEG codec
        '--output', '-'      // Output to stdout
    ]);

    videoProcess.stdout.on('data', (data) => {
        // Convert the frame to base64 and emit to connected clients
        const base64Frame = data.toString('base64');
        socket.emit('videoFrame', base64Frame);
    });

    videoProcess.stderr.on('data', (data) => {
        console.error('Video stream error:', data.toString());
    });

    videoProcess.on('close', (code) => {
        console.log('Video stream process closed with code:', code);
        videoProcess = null;
    });
}

// Function to stop video stream
function stopVideoStream() {
    if (videoProcess) {
        videoProcess.kill();
        videoProcess = null;
        console.log('Video stream stopped');
    }
}

// Function to capture image
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

// Function to remove the captured image
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
const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Existing folder and path definitions
const folderPath = '/home/johannes/tauben/images';
const imagePath = path.join(folderPath, 'test_picture.jpg');

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
    
    // Use libcamera-vid with rotation parameters
    const command = 'libcamera-vid';
    const args = [
        '--codec', 'mjpeg',
        '--width', '640',
        '--height', '480',
        '--framerate', '15',
        '--inline',
        '--nopreview',
        '--rotation', '270',     // Add rotation (try 90, 180, or 270 to find the correct orientation)
        '--timeout', '0',
        '--output', '-'
    ];

    try {
        videoProcess = spawn(command, args);
        let buffer = Buffer.from([]);

        // Handle stdout data (video frames)
        videoProcess.stdout.on('data', (data) => {
            buffer = Buffer.concat([buffer, data]);
            
            // Look for JPEG start and end markers
            let start = 0;
            let end = 0;
            
            while (true) {
                start = buffer.indexOf(Buffer.from([0xFF, 0xD8])); // JPEG start marker
                end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));   // JPEG end marker
                
                if (start !== -1 && end !== -1 && end > start) {
                    const frame = buffer.slice(start, end + 2);
                    // Only emit if the frame is a valid size
                    if (frame.length > 1000) {
                        socket.emit('videoFrame', frame.toString('base64'));
                    }
                    buffer = buffer.slice(end + 2);
                } else {
                    break;
                }
            }
        });

        // Handle stderr (debug messages)
        videoProcess.stderr.on('data', (data) => {
            const message = data.toString();
            if (!message.includes('INFO') && !message.includes('#')) {
                console.log('Video stream message:', message);
            }
        });

        videoProcess.on('close', (code) => {
            console.log(`Video stream process exited with code ${code}`);
            videoProcess = null;
        });

        videoProcess.on('error', (err) => {
            console.error('Video stream process error:', err);
            videoProcess = null;
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

// Update the still image capture to match the rotation
async function captureImage() {
    return new Promise((resolve, reject) => {
        const captureCommand = `libcamera-still -o ${imagePath} -t 1000 --width 1280 --height 720 --rotation 270`;  // Added rotation
        
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
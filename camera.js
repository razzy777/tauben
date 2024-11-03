const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chokidar = require('chokidar');

// Define the folder paths
const folderPath = '/home/johannes/tauben/images';
const streamPath = path.join(folderPath, 'stream');
const currentFrame = path.join(streamPath, 'current.jpg');

// Ensure the directories exist with correct permissions
function ensureDirectories() {
    [folderPath, streamPath].forEach(dir => {
        if (!fs.existsSync(dir)) {
            console.log(`Creating directory at: ${dir}`);
            fs.mkdirSync(dir, { recursive: true });
            fs.chmodSync(dir, 0o777);
        } else {
            console.log(`Directory exists at: ${dir}`);
            fs.chmodSync(dir, 0o777);
        }
    });
}

let streamProcess = null;

async function startVideoStream(socket) {
    if (streamProcess) {
        console.log('Stream already running');
        return;
    }

    ensureDirectories();
    console.log('Starting camera stream...');

    // Using libcamera-vid with MJPEG output
    streamProcess = spawn('libcamera-vid', [
        '--camera', '0',           // First camera
        '--codec', 'mjpeg',        // Use MJPEG codec
        '--width', '640',          // Width
        '--height', '480',         // Height
        '--framerate', '10',       // Reduced framerate for stability
        '--inline',                // Output frames immediately
        '--output', '-',           // Output to stdout
        '--nopreview',             // No preview window
        '--timeout', '0'           // Run indefinitely
    ]);

    let buffer = Buffer.alloc(0);
    const jpegStart = Buffer.from([0xFF, 0xD8]);
    const jpegEnd = Buffer.from([0xFF, 0xD9]);

    streamProcess.stdout.on('data', (data) => {
        buffer = Buffer.concat([buffer, data]);
        
        // Find complete JPEG frames
        let startIndex = 0;
        while (true) {
            const frameStart = buffer.indexOf(jpegStart, startIndex);
            if (frameStart === -1) break;
            
            const frameEnd = buffer.indexOf(jpegEnd, frameStart + 2);
            if (frameEnd === -1) break;
            
            // Extract and emit the frame
            const frame = buffer.slice(frameStart, frameEnd + 2);
            socket.emit('videoFrame', frame.toString('base64'));
            
            startIndex = frameEnd + 2;
        }
        
        // Keep only the incomplete frame data
        if (startIndex > 0) {
            buffer = buffer.slice(startIndex);
        }
    });

    streamProcess.stderr.on('data', (data) => {
        console.error('Stream error:', data.toString());
    });

    streamProcess.on('close', (code) => {
        console.log('Stream process closed with code:', code);
        streamProcess = null;
    });

    streamProcess.on('error', (err) => {
        console.error('Stream process error:', err);
        streamProcess = null;
    });
}

function stopVideoStream() {
    if (streamProcess) {
        streamProcess.kill();
        streamProcess = null;
    }
    console.log('Stream stopped');
}

async function captureImage() {
    return new Promise((resolve, reject) => {
        const imagePath = path.join(folderPath, 'test_picture.jpg');
        const captureCommand = spawn('libcamera-still', [
            '--camera', '0',
            '--width', '1280',
            '--height', '720',
            '--output', imagePath,
            '--nopreview'
        ]);

        captureCommand.stderr.on('data', (data) => {
            console.log('Capture output:', data.toString());
        });

        captureCommand.on('close', (code) => {
            if (code === 0) {
                console.log('Image captured successfully');
                resolve(imagePath);
            } else {
                reject(new Error(`Capture failed with code ${code}`));
            }
        });

        captureCommand.on('error', (err) => {
            reject(new Error(`Capture process error: ${err.message}`));
        });
    });
}

async function removeImage(imagePath) {
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
            resolve(); // Don't reject if file doesn't exist
        }
    });
}

// Function to check if the camera is available
async function checkCamera() {
    return new Promise((resolve) => {
        exec('v4l2-ctl --list-devices', (error, stdout, stderr) => {
            if (error) {
                console.error('Error checking camera:', error);
                resolve(false);
                return;
            }
            
            // Check if PISP devices are present
            const hasPisp = stdout.includes('pispbe');
            console.log('Camera check result:', stdout);
            resolve(hasPisp);
        });
    });
}

module.exports = {
    captureImage,
    removeImage,
    startVideoStream,
    stopVideoStream,
    checkCamera
};
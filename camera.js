const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

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

    // Try to find the correct video device
    const videoDevices = [
        '/dev/video20',  // Try the first PISP device
        '/dev/video21',
        '/dev/video22',
        '/dev/video23'
    ];

    let workingDevice = null;
    for (const device of videoDevices) {
        try {
            // Test if we can capture from this device
            await exec(`v4l2-ctl --device=${device} --all`);
            console.log(`Found working camera device: ${device}`);
            workingDevice = device;
            break;
        } catch (error) {
            console.log(`Device ${device} not suitable:`, error.message);
        }
    }

    if (!workingDevice) {
        console.error('No suitable camera device found');
        return;
    }

    // Use ffmpeg to capture frames from the video device
    streamProcess = spawn('ffmpeg', [
        '-f', 'video4linux2',
        '-input_format', 'mjpeg',  // Try MJPEG first
        '-video_size', '640x480',
        '-i', workingDevice,
        '-vf', 'fps=5',  // Limit to 5 FPS
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-'
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
        console.log('Stream info:', data.toString());
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
    return new Promise(async (resolve, reject) => {
        const imagePath = path.join(folderPath, 'test_picture.jpg');
        
        // Try to find a working video device
        const videoDevices = ['/dev/video20', '/dev/video21', '/dev/video22', '/dev/video23'];
        let workingDevice = null;
        
        for (const device of videoDevices) {
            try {
                await exec(`v4l2-ctl --device=${device} --all`);
                workingDevice = device;
                break;
            } catch (error) {
                console.log(`Device ${device} not suitable for capture`);
            }
        }

        if (!workingDevice) {
            reject(new Error('No suitable camera device found'));
            return;
        }

        // Use ffmpeg to capture a single frame
        const captureProcess = spawn('ffmpeg', [
            '-f', 'video4linux2',
            '-input_format', 'mjpeg',
            '-video_size', '1280x720',
            '-i', workingDevice,
            '-frames:v', '1',
            '-y',  // Overwrite output file
            imagePath
        ]);

        captureProcess.stderr.on('data', (data) => {
            console.log('Capture info:', data.toString());
        });

        captureProcess.on('close', (code) => {
            if (code === 0 && fs.existsSync(imagePath)) {
                resolve(imagePath);
            } else {
                reject(new Error(`Capture failed with code ${code}`));
            }
        });

        captureProcess.on('error', (err) => {
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
            resolve();
        }
    });
}

async function checkCamera() {
    return new Promise(async (resolve) => {
        try {
            // Check for ffmpeg installation
            await exec('which ffmpeg');
            
            // Try to find a working video device
            const videoDevices = ['/dev/video20', '/dev/video21', '/dev/video22', '/dev/video23'];
            for (const device of videoDevices) {
                try {
                    await exec(`v4l2-ctl --device=${device} --all`);
                    console.log(`Found working camera device: ${device}`);
                    resolve(true);
                    return;
                } catch (error) {
                    console.log(`Device ${device} not suitable`);
                }
            }
            resolve(false);
        } catch (error) {
            console.error('Camera check failed:', error);
            resolve(false);
        }
    });
}

module.exports = {
    captureImage,
    removeImage,
    startVideoStream,
    stopVideoStream,
    checkCamera
};
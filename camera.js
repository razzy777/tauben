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

// Function to find a valid capture device
async function findCaptureDevice() {
    return new Promise((resolve) => {
        exec('v4l2-ctl --list-devices', (error, stdout) => {
            if (error) {
                console.error('Error listing devices:', error);
                resolve(null);
                return;
            }

            const devices = [];
            let currentDevice = '';
            const lines = stdout.split('\n');

            for (const line of lines) {
                if (line.includes('/dev/video')) {
                    const device = line.trim();
                    devices.push(device);
                }
            }

            // Test each device for capture capability
            const testDevice = async (index) => {
                if (index >= devices.length) {
                    resolve(null);
                    return;
                }

                const device = devices[index];
                exec(`v4l2-ctl --device=${device} --all`, async (err, stdout) => {
                    if (!err && stdout.includes('Video Capture')) {
                        // Additional verification
                        try {
                            const caps = await new Promise((res) => {
                                exec(`v4l2-ctl --device=${device} --list-formats-ext`, (e, out) => {
                                    if (e) res(false);
                                    else res(out);
                                });
                            });
                            
                            if (caps && (caps.includes('MJPG') || caps.includes('YUYV'))) {
                                console.log(`Found valid capture device: ${device}`);
                                resolve(device);
                                return;
                            }
                        } catch (e) {
                            console.log(`Error checking capabilities for ${device}:`, e);
                        }
                    }
                    testDevice(index + 1);
                });
            };

            testDevice(0);
        });
    });
}

async function startVideoStream(socket) {
    if (streamProcess) {
        console.log('Stream already running');
        return;
    }

    ensureDirectories();
    console.log('Starting camera stream...');

    const device = await findCaptureDevice();
    if (!device) {
        console.error('No suitable camera device found');
        return;
    }

    console.log(`Using camera device: ${device}`);

    // Get supported formats
    const formats = await new Promise((resolve) => {
        exec(`v4l2-ctl --device=${device} --list-formats-ext`, (error, stdout) => {
            resolve(stdout);
        });
    });

    // Determine the best format to use
    const useYUYV = formats.includes('YUYV');
    const inputFormat = useYUYV ? 'yuyv422' : 'mjpeg';
    
    const ffmpegArgs = [
        '-f', 'video4linux2',
        '-input_format', inputFormat,
        '-video_size', '640x480',
        '-i', device
    ];

    if (useYUYV) {
        ffmpegArgs.push(
            '-vf', 'fps=5',  // Limit framerate for YUYV
            '-pix_fmt', 'yuv420p'  // Convert to common format
        );
    }

    ffmpegArgs.push(
        '-f', 'mpegts',
        '-codec:v', 'mpeg1video',
        '-b:v', '800k',
        '-bf', '0',
        '-'
    );

    streamProcess = spawn('ffmpeg', ffmpegArgs);

    streamProcess.stdout.on('data', (data) => {
        socket.emit('videoData', data);
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
        const device = await findCaptureDevice();
        if (!device) {
            reject(new Error('No suitable camera device found'));
            return;
        }

        const imagePath = path.join(folderPath, 'test_picture.jpg');
        
        const captureProcess = spawn('ffmpeg', [
            '-f', 'video4linux2',
            '-i', device,
            '-frames:v', '1',
            '-y',
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
    const device = await findCaptureDevice();
    return !!device;
}

module.exports = {
    captureImage,
    removeImage,
    startVideoStream,
    stopVideoStream,
    checkCamera
};
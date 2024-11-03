const { spawn } = require('child_process');
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

function startVideoStream(socket) {
    if (streamProcess) {
        console.log('Stream already running');
        return;
    }

    ensureDirectories();
    console.log('Starting camera stream...');

    // Try different approaches in sequence
    tryStreamMethods(socket, 0);
}

function tryStreamMethods(socket, methodIndex) {
    const streamMethods = [
        // Method 1: libcamera-vid
        {
            command: 'libcamera-vid',
            args: [
                '--inline',
                '--width', '640',
                '--height', '480',
                '--framerate', '15',
                '--codec', 'mjpeg',
                '--output', '-'
            ]
        },
        // Method 2: raspi-still in timelapse mode
        {
            command: 'raspistill',
            args: [
                '-w', '640',
                '-h', '480',
                '-q', '10',
                '-tl', '200',
                '-t', '0',
                '-o', currentFrame,
                '-n'
            ]
        }
    ];

    if (methodIndex >= streamMethods.length) {
        console.error('All streaming methods failed');
        return;
    }

    const method = streamMethods[methodIndex];
    console.log(`Trying streaming method ${methodIndex + 1} using ${method.command}`);

    try {
        streamProcess = spawn(method.command, method.args);
        let frameBuffer = Buffer.alloc(0);

        streamProcess.stdout?.on('data', (data) => {
            if (method.command === 'libcamera-vid') {
                frameBuffer = Buffer.concat([frameBuffer, data]);
                
                // Look for JPEG markers
                while (frameBuffer.length > 2) {
                    const startIndex = frameBuffer.indexOf(Buffer.from([0xff, 0xd8]));
                    const endIndex = frameBuffer.indexOf(Buffer.from([0xff, 0xd9]));
                    
                    if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
                        const frame = frameBuffer.slice(startIndex, endIndex + 2);
                        socket.emit('videoFrame', frame.toString('base64'));
                        frameBuffer = frameBuffer.slice(endIndex + 2);
                    } else {
                        break;
                    }
                }
            }
        });

        if (method.command === 'raspistill') {
            // Watch the output file for changes
            const watcher = fs.watch(currentFrame, (eventType) => {
                if (eventType === 'change') {
                    try {
                        const frame = fs.readFileSync(currentFrame);
                        socket.emit('videoFrame', frame.toString('base64'));
                    } catch (error) {
                        // Ignore read errors during file writes
                    }
                }
            });

            // Store watcher reference for cleanup
            streamProcess.watcher = watcher;
        }

        streamProcess.stderr?.on('data', (data) => {
            console.log('Stream output:', data.toString());
        });

        streamProcess.on('error', (error) => {
            console.error(`Failed to start ${method.command}:`, error);
            streamProcess = null;
            // Try next method
            tryStreamMethods(socket, methodIndex + 1);
        });

        streamProcess.on('close', (code) => {
            console.log(`${method.command} process closed with code ${code}`);
            if (streamProcess?.watcher) {
                streamProcess.watcher.close();
            }
            streamProcess = null;
            
            // If process exits immediately, try next method
            if (code !== 0) {
                tryStreamMethods(socket, methodIndex + 1);
            }
        });
    } catch (error) {
        console.error(`Error starting ${method.command}:`, error);
        tryStreamMethods(socket, methodIndex + 1);
    }
}

function stopVideoStream() {
    if (streamProcess) {
        if (streamProcess.watcher) {
            streamProcess.watcher.close();
        }
        streamProcess.kill();
        streamProcess = null;
    }
    console.log('Stream stopped');
}

async function captureImage() {
    return new Promise((resolve, reject) => {
        const imagePath = path.join(folderPath, 'test_picture.jpg');
        
        // Try libcamera-still first
        const captureProcess = spawn('libcamera-still', [
            '-o', imagePath,
            '--width', '1280',
            '--height', '720',
            '--nopreview'
        ]);

        captureProcess.stderr.on('data', (data) => {
            console.log('Capture output:', data.toString());
        });

        captureProcess.on('close', (code) => {
            if (code === 0 && fs.existsSync(imagePath)) {
                resolve(imagePath);
            } else {
                // If libcamera-still fails, try raspistill
                const raspistillProcess = spawn('raspistill', [
                    '-o', imagePath,
                    '-w', '1280',
                    '-h', '720',
                    '-n'
                ]);

                raspistillProcess.on('close', (code2) => {
                    if (code2 === 0 && fs.existsSync(imagePath)) {
                        resolve(imagePath);
                    } else {
                        reject(new Error('Both capture methods failed'));
                    }
                });
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
    return new Promise((resolve) => {
        // Try a test capture
        const testProcess = spawn('libcamera-still', ['--list-cameras']);
        
        testProcess.on('close', (code) => {
            resolve(code === 0);
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
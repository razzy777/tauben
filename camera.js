const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chokidar = require('chokidar'); // Add this to your package.json

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

// Clean up any existing stream files
function cleanupStreamFiles() {
    if (fs.existsSync(currentFrame)) {
        fs.unlinkSync(currentFrame);
    }
}

let streamProcess = null;
let watcher = null;

function startVideoStream(socket) {
    if (streamProcess) {
        console.log('Stream already running');
        return;
    }

    ensureDirectories();
    cleanupStreamFiles();

    console.log('Starting camera stream...');

    // Start raspistill in timelapse mode
    streamProcess = spawn('raspistill', [
        '-w', '640',          // Width
        '-h', '480',          // Height
        '-q', '10',           // Quality (lower number = higher quality)
        '-o', currentFrame,   // Output file
        '-tl', '200',         // Time between shots (ms)
        '-t', '0',            // Run indefinitely
        '-s',                 // No preview window
        '-n'                  // No preview window
    ]);

    streamProcess.stderr.on('data', (data) => {
        console.log('Stream output:', data.toString());
    });

    streamProcess.on('close', (code) => {
        console.log('Stream process closed with code:', code);
        streamProcess = null;
    });

    // Watch for file changes and emit to socket
    watcher = chokidar.watch(currentFrame, {
        persistent: true,
        awaitWriteFinish: {
            stabilityThreshold: 100,
            pollInterval: 100
        }
    });

    watcher.on('change', (path) => {
        try {
            const imageData = fs.readFileSync(currentFrame);
            const base64Image = imageData.toString('base64');
            socket.emit('videoFrame', base64Image);
        } catch (error) {
            console.error('Error reading frame:', error);
        }
    });
}

function stopVideoStream() {
    if (streamProcess) {
        streamProcess.kill();
        streamProcess = null;
    }
    
    if (watcher) {
        watcher.close();
        watcher = null;
    }

    cleanupStreamFiles();
    console.log('Stream stopped');
}

// Original photo capture function
async function captureImage() {
    return new Promise((resolve, reject) => {
        const imagePath = path.join(folderPath, 'test_picture.jpg');
        const captureCommand = `raspistill -o ${imagePath} -t 1000 --width 1280 --height 720`;

        exec(captureCommand, (err, stdout, stderr) => {
            if (err) {
                reject(new Error(`Error capturing image: ${err.message}`));
                return;
            }
            if (stderr) {
                console.error('raspistill error:', stderr);
            }
            console.log('Image successfully captured and saved at:', imagePath);
            resolve(imagePath);
        });
    });
}

// Original remove image function
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
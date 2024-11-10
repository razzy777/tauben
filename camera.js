// camera.js

const { spawn } = require('child_process');

let videoProcess = null;

function startVideoStream(frontendNamespace, aiNamespace) {
  if (videoProcess) {
    console.log('Video stream already running');
    return;
  }

  console.log('Starting video stream...');

  const command = 'libcamera-vid';
  const args = [
    '--codec', 'mjpeg',
    '--width', '1920',
    '--height', '1080',
    '--framerate', '15',
    '--inline',
    '--nopreview',
    '--timeout', '0',
    '--output', '-'
  ];

  try {
    videoProcess = spawn(command, args);
    let buffer = Buffer.from([]);

    // Variables to handle AI frame interval
    let lastAIFrameTime = 0;
    const aiFrameRate = 1; // Send frame to AI processor every 1 second

    videoProcess.stdout.on('data', (data) => {
      buffer = Buffer.concat([buffer, data]);

      let start = 0;
      let end = 0;

      while (true) {
        start = buffer.indexOf(Buffer.from([0xFF, 0xD8]));
        end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));

        if (start !== -1 && end !== -1 && end > start) {
          const frame = buffer.slice(start, end + 2);
          if (frame.length > 1000) {
            // Emit video frame to frontend clients
            frontendNamespace.emit('videoFrame', frame.toString('base64'));

            // Send frame to AI processor at specified interval
            const currentTime = Date.now();
            if (currentTime - lastAIFrameTime >= aiFrameRate * 1000) {
              aiNamespace.emit('videoFrame', frame);
              console.log('Sent frame to AI processor'); // Add logging
              lastAIFrameTime = currentTime;
            }
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

function stopVideoStream() {
  if (videoProcess) {
    videoProcess.kill('SIGTERM');
    videoProcess = null;
    console.log('Video stream stopped');
  }
}

module.exports = {
  startVideoStream,
  stopVideoStream
};

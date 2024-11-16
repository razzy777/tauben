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
    '--width', '640',
    '--height', '480',
    '--framerate', '30',  // Updated to match your working test
    '--timeout', '0',
    '--output', '-',
    '--nopreview'
  ];

  console.log('Executing command:', command, args.join(' '));

  try {
    videoProcess = spawn(command, args);
    console.log('Video process started with PID:', videoProcess.pid);

    let buffer = Buffer.from([]);
    let frameCount = 0;
    let lastFrameTime = Date.now();

    videoProcess.stdout.on('data', (data) => {
      try {
        buffer = Buffer.concat([buffer, data]);
        
        let start = 0;
        let end = 0;

        while (true) {
          start = buffer.indexOf(Buffer.from([0xFF, 0xD8]));
          end = buffer.indexOf(Buffer.from([0xFF, 0xD9]));

          if (start !== -1 && end !== -1 && end > start) {
            const frame = buffer.slice(start, end + 2);
            frameCount++;
            
            if (frame.length > 1000) {
              const currentTime = Date.now();
              const fps = 1000 / (currentTime - lastFrameTime);
              lastFrameTime = currentTime;

              // Log every 100th frame to avoid console spam
              if (frameCount % 100 === 0) {
                console.log(`Frame ${frameCount}, Size: ${frame.length}, FPS: ${fps.toFixed(2)}`);
              }

              // Send to frontend
              frontendNamespace.emit('videoFrame', frame.toString('base64'));

              // Send to AI processor at a lower rate (every 500ms)
              if (frameCount % 15 === 0) {  // At 30fps, this is every 500ms
                const aiClients = Object.keys(aiNamespace.sockets).length;
                if (aiClients > 0) {
                  console.log('Sending frame to AI processor');
                  aiNamespace.emit('videoFrame', frame.toString('base64'));
                }
              }
            }
            
            buffer = buffer.slice(end + 2);
          } else {
            break;
          }
        }
      } catch (error) {
        console.error('Error processing video frame:', error);
      }
    });

    videoProcess.stderr.on('data', (data) => {
      console.log('Camera stderr:', data.toString());
    });

    videoProcess.on('error', (error) => {
      console.error('Camera process error:', error);
    });

    videoProcess.on('exit', (code, signal) => {
      console.log('Camera process exited with code:', code, 'signal:', signal);
      videoProcess = null;
    });

  } catch (error) {
    console.error('Failed to start video stream:', error);
    videoProcess = null;
  }
}

function stopVideoStream() {
  if (videoProcess) {
    console.log('Stopping video stream...');
    videoProcess.kill('SIGTERM');
    videoProcess = null;
    console.log('Video stream stopped');
  }
}

module.exports = {
  startVideoStream,
  stopVideoStream
};
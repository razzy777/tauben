// camera.js
const { spawn } = require('child_process');

let videoProcess = null;

function startVideoStream(frontendNamespace, aiNamespace) {
  if (videoProcess) {
    console.log('Video stream already running');
    return;
  }

  console.log('Starting video stream...');
  console.log('Frontend namespace clients:', frontendNamespace.sockets.size);
  console.log('AI namespace clients:', aiNamespace.sockets.size);

  const command = 'libcamera-vid';
  const args = [
    '--codec', 'mjpeg',
    '--width', '640',
    '--height', '480',
    '--framerate', '30',
    '--inline',  // Added inline flag
    '--nopreview',
    '--timeout', '0',
    '--output', '-'
  ];

  console.log('Executing command:', command, args.join(' '));

  try {
    // Check if libcamera-vid exists
    const { execSync } = require('child_process');
    try {
      execSync('which libcamera-vid');
      console.log('libcamera-vid found');
    } catch (error) {
      console.error('libcamera-vid not found in PATH');
      return;
    }

    // Start video process
    videoProcess = spawn(command, args);
    console.log('Video process started with PID:', videoProcess.pid);

    let buffer = Buffer.from([]);
    let frameCount = 0;

    // Handle stdout data
    videoProcess.stdout.on('data', (data) => {
      try {
        if (frameCount === 0) {
          console.log('First video data received');
        }
        
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
              if (frameCount % 30 === 0) {  // Log every 30 frames
                console.log(`Frame ${frameCount}, Size: ${frame.length} bytes`);
              }

              // Send to frontend
              const frontendClients = frontendNamespace.sockets.size;
              if (frontendClients > 0) {
                frontendNamespace.emit('videoFrame', frame.toString('base64'));
                if (frameCount % 30 === 0) {
                  console.log('Frame sent to frontend clients:', frontendClients);
                }
              }

              // Send to AI (every 15th frame)
              const aiClients = aiNamespace.sockets.size;
              if (aiClients > 0 && frameCount % 15 === 0) {
                aiNamespace.emit('videoFrame', frame.toString('base64'));
                console.log('Frame sent to AI clients:', aiClients);
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

    // Handle stderr (camera info and errors)
    videoProcess.stderr.on('data', (data) => {
      console.log('Camera stderr:', data.toString());
    });

    // Handle process errors
    videoProcess.on('error', (error) => {
      console.error('Camera process error:', error);
      videoProcess = null;
    });

    // Handle process exit
    videoProcess.on('exit', (code, signal) => {
      console.log('Camera process exited with code:', code, 'signal:', signal);
      videoProcess = null;
    });

  } catch (error) {
    console.error('Failed to start video stream:', error);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
    videoProcess = null;
  }
}

function stopVideoStream() {
  if (videoProcess) {
    console.log('Stopping video stream...');
    try {
      videoProcess.kill('SIGTERM');
      console.log('Video process terminated');
    } catch (error) {
      console.error('Error stopping video stream:', error);
    }
    videoProcess = null;
  } else {
    console.log('No video stream to stop');
  }
}

module.exports = {
  startVideoStream,
  stopVideoStream
};
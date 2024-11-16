const { spawn } = require('child_process');

let videoProcess = null;

// Add camera test function
async function testCamera() {
  return new Promise((resolve, reject) => {
    console.log('Testing camera availability...');
    const test = spawn('libcamera-still', ['--immediate', '-o', '/dev/null', '-t', '1']);
    
    test.on('error', (error) => {
      console.error('Camera test error:', error);
      reject(error);
    });
    
    test.on('exit', (code) => {
      if (code === 0) {
        console.log('Camera test successful');
        resolve(true);
      } else {
        console.error('Camera test failed with code:', code);
        reject(new Error(`Camera test failed with code ${code}`));
      }
    });
  });
}

async function startVideoStream(frontendNamespace, aiNamespace) {
  if (videoProcess) {
    console.log('Video stream already running');
    return;
  }

  try {
    // Test camera before starting stream
    await testCamera();
    
    console.log('Starting video stream...');

    const command = 'libcamera-vid';
    const args = [
      '--codec', 'mjpeg',
      '--width', '640',
      '--height', '480',
      '--framerate', '30',
      '--timeout', '0',
      '--output', '-',
      '--nopreview',
      '--verbose', '1'  // Add verbose output for debugging
    ];

    console.log('Executing command:', command, args.join(' '));

    videoProcess = spawn(command, args);
    console.log('Video process started with PID:', videoProcess.pid);

    // Add error handler for spawn
    videoProcess.on('error', (error) => {
      console.error('Failed to start video process:', error);
      videoProcess = null;
      throw error;
    });

    let buffer = Buffer.from([]);
    let frameCount = 0;
    let lastFrameTime = Date.now();
    let noDataTimeout = null;

    // Add timeout to detect if no data is received
    const resetNoDataTimeout = () => {
      if (noDataTimeout) clearTimeout(noDataTimeout);
      noDataTimeout = setTimeout(() => {
        console.error('No video data received for 5 seconds, restarting stream...');
        stopVideoStream();
        startVideoStream(frontendNamespace, aiNamespace);
      }, 5000);
    };

    resetNoDataTimeout();

    videoProcess.stdout.on('data', (data) => {
      try {
        resetNoDataTimeout();
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

              // Log more frequently during initial startup
              if (frameCount <= 10 || frameCount % 100 === 0) {
                console.log(`Frame ${frameCount}, Size: ${frame.length}, FPS: ${fps.toFixed(2)}`);
              }

              frontendNamespace.emit('videoFrame', frame.toString('base64'));

              if (frameCount % 15 === 0) {
                const aiClients = aiNamespace.sockets.size;
                if (aiClients > 0) {
                  console.log(`Sending frame to ${aiClients} AI client(s)`);
                  aiNamespace.emit('videoFrame', frame.toString('base64'));
                }
              }
            } else {
              console.warn(`Received small frame (${frame.length} bytes), skipping`);
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
      const message = data.toString();
      console.log('Camera stderr:', message);
      
      // Check for specific error messages
      if (message.includes('Permission denied') || 
          message.includes('Device or resource busy')) {
        console.error('Camera access error detected, attempting restart...');
        stopVideoStream();
        setTimeout(() => startVideoStream(frontendNamespace, aiNamespace), 1000);
      }
    });

    videoProcess.on('exit', (code, signal) => {
      console.log('Camera process exited with code:', code, 'signal:', signal);
      if (noDataTimeout) clearTimeout(noDataTimeout);
      videoProcess = null;
      
      // Attempt restart if exit was unexpected
      if (code !== 0 && signal !== 'SIGTERM') {
        console.log('Unexpected camera exit, attempting restart in 2 seconds...');
        setTimeout(() => startVideoStream(frontendNamespace, aiNamespace), 2000);
      }
    });

  } catch (error) {
    console.error('Failed to start video stream:', error);
    videoProcess = null;
    throw error;
  }
}

function stopVideoStream() {
  if (videoProcess) {
    console.log('Stopping video stream...');
    try {
      videoProcess.kill('SIGTERM');
    } catch (error) {
      console.error('Error stopping video stream:', error);
      // Force kill if SIGTERM fails
      try {
        videoProcess.kill('SIGKILL');
      } catch (secondError) {
        console.error('Failed to force kill video process:', secondError);
      }
    }
    videoProcess = null;
    console.log('Video stream stopped');
  }
}

module.exports = {
  startVideoStream,
  stopVideoStream,
  testCamera
};
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const { spawn } = require('child_process');
const { ServoController } = require('./relay');
const servoSystem = require('./servoSystem');


// Create the server
const server = http.createServer();

// Create Socket.IO server with optimized settings
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  },
  transports: ['websocket'],
  upgrade: false,
  maxHttpBufferSize: 64000,
  pingTimeout: 10000,
  perMessageDeflate: false
});

// Create namespaces
const frontendNamespace = io.of('/frontend');
const aiNamespace = io.of('/ai');

// Global variables
let videoProcess = null;
let isStreamActive = false;

// Initialize servo controller
const relayController = new ServoController(588);

// Video stream handling
function startVideoStream(frontendNamespace, aiNamespace) {
  if (videoProcess) {
    console.log('Video stream already running');
    return;
  }

  console.log('Starting video stream...');

  const args = [
    '--codec', 'mjpeg',
    '--width', '320',
    '--height', '240',
    '--framerate', '10',
    '--timeout', '0',
    '--output', '-',
    '--nopreview'
  ];

  try {
    videoProcess = spawn('libcamera-vid', args);
    console.log('Video process started with PID:', videoProcess.pid);

    let frameBuffer = Buffer.from([]);
    let lastFrameTime = Date.now();
    const FRAME_INTERVAL = 200; // 200ms between frames

    videoProcess.stdout.on('data', (data) => {
      const now = Date.now();
      if (now - lastFrameTime >= FRAME_INTERVAL) {
        try {
          frameBuffer = Buffer.concat([frameBuffer, data]);
          
          let start = 0;
          let end = 0;

          while (true) {
            start = frameBuffer.indexOf(Buffer.from([0xFF, 0xD8]));
            end = frameBuffer.indexOf(Buffer.from([0xFF, 0xD9]));

            if (start !== -1 && end !== -1 && end > start) {
              const frame = frameBuffer.slice(start, end + 2);
              
              if (frame.length > 1000 && frame.length < 64000) {
                // Only emit if we have connected clients
                if (frontendNamespace.sockets.size > 0) {
                  frontendNamespace.emit('videoFrame', frame.toString('base64'));
                }
                
                // Send to AI clients at a lower rate
                if (aiNamespace.sockets.size > 0 && now % 500 === 0) {
                  aiNamespace.emit('videoFrame', frame.toString('base64'));
                }
                
                lastFrameTime = now;
              }
              
              frameBuffer = frameBuffer.slice(end + 2);
            } else {
              break;
            }
          }
        } catch (error) {
          console.error('Error processing video frame:', error);
        }
      }
    });

    videoProcess.stderr.on('data', (data) => {
      console.log('Camera stderr:', data.toString());
    });

    videoProcess.on('error', (error) => {
      console.error('Video process error:', error);
      stopVideoStream();
    });

    videoProcess.on('exit', (code, signal) => {
      console.log('Video process exited with code:', code, 'signal:', signal);
      videoProcess = null;
      isStreamActive = false;
    });

    isStreamActive = true;

  } catch (error) {
    console.error('Failed to start video stream:', error);
    videoProcess = null;
    isStreamActive = false;
  }
}

function stopVideoStream() {
  if (videoProcess) {
    console.log('Stopping video stream...');
    try {
      videoProcess.kill('SIGTERM');
      setTimeout(() => {
        if (videoProcess) {
          videoProcess.kill('SIGKILL');
        }
      }, 1000);
    } catch (error) {
      console.error('Error stopping video stream:', error);
    }
    videoProcess = null;
    isStreamActive = false;
  }
}

// Connection handling for frontend clients
frontendNamespace.on('connection', (socket) => {
  console.log('Frontend client connected:', socket.id);
  
  // Start video stream if not already running
  if (!isStreamActive) {
    startVideoStream(frontendNamespace, aiNamespace);
  }

  // Handle servo movement
  socket.on('moveServoRelative', ({ pan, tilt }) => {
    try {
        console.log(`Moving servo relatively: pan=${pan}, tilt=${tilt}`);
        this.servoSystem.moveRelative(pan, tilt);
    } catch (error) {
      console.error('Error moving servo:', error);
    }
  });

  // Handle water activation
  socket.on('activateWater', async (duration) => {
    try {
      console.log(`Activating water for ${duration}ms`);
      await this.servoSystem.activateWater(duration);
    } catch (error) {
      console.error('Error activating water:', error);
    }
  });

  // Handle center command
  socket.on('centerServo', () => {
    try {
      console.log('Centering servos');
      this.servoSystem.center();
    } catch (error) {
      console.error('Error centering servos:', error);
    }
  });

  // Handle scan command
  socket.on('startScan', async () => {
    try {
      console.log('Starting scan');
      await this.servoSystem.scan();
    } catch (error) {
      console.error('Error during scan:', error);
    }
  });

  // Handle disconnection
  socket.on('disconnect', () => {
    console.log('Frontend client disconnected:', socket.id);
    
    // Stop video stream if no clients connected
    if (frontendNamespace.sockets.size === 0 && aiNamespace.sockets.size === 0) {
      stopVideoStream();
    }
  });
});

// Connection handling for AI clients
aiNamespace.on('connection', (socket) => {
  console.log('AI client connected:', socket.id);

  socket.on('detections', (detections) => {
    // Forward detections to frontend clients
    frontendNamespace.emit('detections', detections);
  });

  socket.on('disconnect', () => {
    console.log('AI client disconnected:', socket.id);
  });
});

// Error handling
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Graceful shutdown
async function handleShutdown() {
  console.log('Shutting down...');
  
  stopVideoStream();
  
  try {
    await this.servoSystem.cleanup();
    console.log('Relay controller cleaned up');
    
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
}

// Handle termination signals
process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
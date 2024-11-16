// server.js
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const { startVideoStream, stopVideoStream } = require('./camera');
const servoSystem = require('./servoSystem');
const { ServoController } = require('./relay');

// Create the relay controller
const relayController = new ServoController(588); // Replace with the appropriate pin number

const server = http.createServer();

// Create a single Socket.IO server
const io = socketIo(server, {
  cors: {
    origin: "*", // Replace with your frontend domain in production
    methods: ["GET", "POST"]
  }
});

// Create namespaces
const frontendNamespace = io.of('/frontend');
console.log('Frontend namespace created');

const aiNamespace = io.of('/ai');
console.log('AI namespace created');

// Flag to prevent multiple video streams
let videoStreamStarted = false;

// Initialize system components
async function initializeSystem() {
  try {
    console.log('Initializing system components...');

    // Initialize servo system
    await servoSystem.initialize();
    console.log('Servo system initialized');

    // Initialize relay controller
    await relayController.init();
    console.log('Relay controller initialized');

    // Start the server
    server.listen(3000, () => {
      console.log('Socket server running on port 3000');
      console.log('System initialization complete, ready for commands');
    });
  } catch (error) {
    console.error('System initialization failed:', error);
    process.exit(1);
  }
}

// Adjust servos to follow detected person
function adjustServosToFollow(boundingBox) {
  const [ymin, xmin, ymax, xmax] = boundingBox;
  const centerX = (xmin + xmax) / 2;
  const centerY = (ymin + ymax) / 2;

  // Map centerX and centerY to servo pulse deltas
  const { panDelta, tiltDelta } = mapBoundingBoxToServoDelta(centerX, centerY);

  // Move servos relatively
  servoSystem.moveToPositionRelative(panDelta, tiltDelta);
  
}


function mapBoundingBoxToServoDelta(centerX, centerY) {
  // Constants for maximum step sizes (adjust as needed)
  const MAX_PAN_DELTA = 10;   // Maximum pulse delta for pan
  const MAX_TILT_DELTA = 10;  // Maximum pulse delta for tilt

  // Calculate deviations from the center (normalized between -0.5 and 0.5)
  const deltaX = centerX - 0.5; // Positive if object is to the right
  const deltaY = centerY - 0.5; // Positive if object is below the center

  // Multiply by 2 to get range from -1 to 1
  const normalizedDeltaX = deltaX * 2;
  const normalizedDeltaY = deltaY * 2;

  // Calculate pulse deltas proportional to the normalized deviations
  const panDelta = normalizedDeltaX * MAX_PAN_DELTA;
  const tiltDelta = -normalizedDeltaY * MAX_TILT_DELTA; // Negative to adjust for coordinate system

  return { panDelta, tiltDelta };
}

// Perform initial servo movement test
async function performInitialServoTest() {
  console.log('Performing initial servo test...');

  // Center the servos
  servoSystem.centerServos();
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Test extreme positions
  const testPositions = [
    // Test maximum ranges
    { pan: servoSystem.PAN_MAX_RIGHT_PULSE, tilt: servoSystem.TILT_MAX_UP_PULSE },
    { pan: servoSystem.PAN_MAX_LEFT_PULSE, tilt: servoSystem.TILT_MAX_DOWN_PULSE },
    { pan: servoSystem.PAN_MAX_RIGHT_PULSE, tilt: servoSystem.TILT_MAX_DOWN_PULSE },
    { pan: servoSystem.PAN_MAX_LEFT_PULSE, tilt: servoSystem.TILT_MAX_UP_PULSE },
    // Return to center
    { pan: servoSystem.PAN_CENTER_PULSE, tilt: servoSystem.TILT_CENTER_PULSE }
  ];

  for (const position of testPositions) {
    servoSystem.moveToPosition(position.pan, position.tilt);
    await new Promise(resolve => setTimeout(resolve, 1500));
  }

  console.log('Initial servo test completed');
}

async function performScan() {
  // Define scanning parameters
  const panStart = servoSystem.PAN_MAX_RIGHT_PULSE;
  const panEnd = servoSystem.PAN_MAX_LEFT_PULSE;
  const tiltStart = servoSystem.TILT_MAX_DOWN_PULSE;
  const tiltEnd = servoSystem.TILT_MAX_UP_PULSE;

  const panSteps = 5;
  const tiltSteps = 3;
  const delayBetweenMoves = 1000; // milliseconds

  const panStepSize = (panEnd - panStart) / panSteps;
  const tiltStepSize = (tiltEnd - tiltStart) / tiltSteps;

  for (let tiltPulse = tiltStart; tiltPulse <= tiltEnd; tiltPulse += tiltStepSize) {
    for (let panPulse = panStart; panPulse <= panEnd; panPulse += panStepSize) {
      await servoSystem.moveToPositionAndWait(panPulse, tiltPulse);
      await new Promise(resolve => setTimeout(resolve, delayBetweenMoves));
    }
  }

  // Return to center position
  await servoSystem.moveToPositionAndWait(servoSystem.PAN_CENTER_PULSE, servoSystem.TILT_CENTER_PULSE);
}

// Socket connection handler for frontend clients
frontendNamespace.on('connection', (socket) => {
    console.log('Frontend client connected:', socket.id);
    console.log('Total frontend clients:', frontendNamespace.sockets.size);

    if (!videoStreamStarted) {
        console.log('Starting video stream for new frontend connection');
        startVideoStream(frontendNamespace, aiNamespace);
        videoStreamStarted = true;
    }
  // Handle photo capture requests
  socket.on('takePhoto', async () => {
    try {
      console.log('Taking photo...');
      const imagePath = await captureImage();
      const imageData = fs.readFileSync(imagePath);
      const base64Image = imageData.toString('base64');

      socket.emit('detection', {
        detected: true,
        timestamp: Date.now(),
        image: base64Image,
      });

      console.log('Photo taken and sent to frontend');
      await removeImage(imagePath);
    } catch (error) {
      console.error('Error taking photo:', error);
      socket.emit('error', { message: 'Failed to take photo' });
    }
  });

  // Handle absolute servo movement
  socket.on('moveServo', ({ pan, tilt }) => {
    try {
      console.log(`Moving servo to position: pan=${pan}, tilt=${tilt}`);
      servoSystem.moveToPosition(pan, tilt);
      socket.emit('servoMoved', { success: true });
    } catch (error) {
      console.error('Error moving servo:', error);
      socket.emit('error', { message: 'Failed to move servo' });
    }
  });

  // Handle relative servo movement
  socket.on('moveServoRelative', ({ pan, tilt }) => {
    try {
      console.log(`Moving servo relatively: pan=${pan}, tilt=${tilt}`);
      servoSystem.moveToPositionRelative(pan, tilt);
      socket.emit('servoMoved', { success: true });
    } catch (error) {
      console.error('Error moving servo relatively:', error);
      socket.emit('error', { message: 'Failed to move servo' });
    }
  });

  // Handle servo centering
  socket.on('centerServo', () => {
    try {
      console.log('Centering servos...');
      servoSystem.centerServos();
      socket.emit('servoCentered', { success: true });
    } catch (error) {
      console.error('Error centering servos:', error);
      socket.emit('error', { message: 'Failed to center servos' });
    }
  });

  // Handle water activation
  socket.on('activateWater', async (duration) => {
    try {
      console.log(`Activating water for ${duration}ms...`);
      await relayController.activateWater(duration);
      socket.emit('waterActivated', { success: true });
      console.log('Water activation completed');
    } catch (error) {
      console.error('Error activating water:', error);
      socket.emit('error', { message: 'Failed to activate water' });
    }
  });

  // Handle scan command
  socket.on('startScan', async () => {
    try {
      console.log('Starting scan...');
      await performScan();
      socket.emit('scanCompleted');
    } catch (error) {
      console.error('Error during scanning:', error);
      socket.emit('error', { message: 'Failed to perform scan' });
    }
  });

  // Handle client disconnect
  socket.on('disconnect', () => {
    console.log('Frontend client disconnected:', socket.id);
    console.log('Remaining frontend clients:', frontendNamespace.sockets.size);

    if (frontendNamespace.sockets.size === 0) {
      console.log('No frontend clients left, stopping video stream');
      stopVideoStream();
      videoStreamStarted = false;
    }
  });


});

// Update the AI connection handler
aiNamespace.on('connection', (socket) => {
    console.log('AI client connected:', socket.id);
    console.log('Total AI clients:', aiNamespace.sockets.size);
    socket.on('aiDetections', (detections) => {
        // Broadcast detections to connected frontend clients
        frontendNamespace.emit('detections', detections);
    
        // Implement logic to move servos based on detections
        if (detections && detections.length > 0) {
          const personDetection = detections[0];  // Using the first detected person
          adjustServosToFollow(personDetection.box);
        }
      });
    
    socket.on('disconnect', () => {
      console.log('AI client disconnected:', socket.id);
      console.log('Remaining AI clients:', aiNamespace.sockets.size);
    });
  });

// Error handling for uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

// Error handling for unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Graceful shutdown handler
async function handleShutdown() {
  console.log('\nShutting down...');

  try {
    // Cleanup servo system
    await servoSystem.cleanup();
    console.log('Servo system cleaned up');

    // Cleanup relay controller
    await relayController.cleanup();
    console.log('Relay controller cleaned up');

    // Close server
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

// Start the system
initializeSystem().catch(error => {
  console.error('Failed to start system:', error);
  process.exit(1);
});

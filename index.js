const http = require('http');
const socketIo = require('socket.io');
const servoSystem = require('./servoSystem');
const { captureImage, removeImage, startVideoStream, stopVideoStream } = require('./camera');

const { ServoController } = require('./relay');
const fs = require('fs');


// Create the relay controller
const relayController = new ServoController(588); // Replace with appropriate pin number

// Create server
const server = http.createServer();
const io = socketIo(server, {
  cors: {
    origin: "*", // Replace with your frontend domain in production
    methods: ["GET", "POST"]
  }
});

async function initializeSystem() {
    try {
      console.log('Initializing system components...');
      
      // Check camera availability
      const cameraAvailable = false
      if (!cameraAvailable) {
        console.error('No camera available!');
        // You can choose to continue without camera or exit
        // process.exit(1);
      } else {
        console.log('Camera detected and available');
      }
  
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

// Socket connection handler
function handleSocketConnection(socket) {
  console.log('New client connected:', socket.id);
  startVideoStream(socket);


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
  // Add this to your socket connection handler
    socket.on('getServoStatus', () => {
        try {
        const status = servoSystem.getQueueStatus();
        socket.emit('servoStatus', status);
        } catch (error) {
        console.error('Error getting servo status:', error);
        socket.emit('error', { message: 'Failed to get servo status' });
        }
    });
  
  // Update your move handlers to include status updates
  socket.on('moveServoRelative', ({ pan, tilt }) => {
    try {
      console.log(`Moving servo relatively: pan=${pan}, tilt=${tilt}`);
      servoSystem.moveToPositionRelative(pan, tilt);
      
      // Send immediate status update
      const status = servoSystem.getQueueStatus();
      socket.emit('servoStatus', status);
      
    } catch (error) {
      console.error('Error moving servo relatively:', error);
      socket.emit('error', { message: 'Failed to move servo' });
    }
  });

  // Handle client disconnect
  socket.on('disconnect', () => {
    stopVideoStream(); // Stop video stream when client disconnects
    console.log('Client disconnected:', socket.id);
  });
}

// Set up socket connection handling
io.on('connection', handleSocketConnection);

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
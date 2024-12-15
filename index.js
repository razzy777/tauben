// server.js
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const { startVideoStream, stopVideoStream } = require('./camera');
const servoSystem = require('./servoSystem');
const { ServoController } = require('./relay');
const cocoLabels = fs.readFileSync('coco.txt', 'utf-8').split('\n');


// Create the relay controller
//

// 588 = Channel 1 (Solenoid)
// 598 = Channel 2 (Pump)
const relayControllerSolenoid = new ServoController(588); // Replace with the appropriate pin number
const relayControllerPump = new ServoController(598); // Replace with the appropriate pin number


const server = http.createServer();

// Create a single Socket.IO server
const io = socketIo(server, {
  cors: {
    origin: "*", // Replace with your frontend domain in production
    methods: ["GET", "POST"]
  }, 
  transports: ['websocket'],
  upgrade: false,
  maxHttpBufferSize: 64000,
  pingTimeout: 10000,
  perMessageDeflate: false

});
let timer = 0

// Create namespaces
const frontendNamespace = io.of('/frontend');
const aiNamespace = io.of('/ai');

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
    await relayControllerSolenoid.init();
    console.log('relayControllerSolenoid controller initialized');

    // Initialize relay controller
    await relayControllerPump.init();
    console.log('relayControllerPump controller initialized');
    

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
    //box=[0.703125, 0.3421875, 0.75, 0.390625]
    // float(ymin) / h,
    // float(xmin) / w,
    // float(ymax) / h,
    // float(xmax) / w
  const [ymin, xmin, ymax, xmax] = boundingBox;
  const centerX = (xmin + xmax) / 2;
  const centerY = (ymin + ymax) / 2;
    console.log('CENTER X', centerX)
    console.log('CENTER Y', centerY)

  // Map centerX and centerY to servo pulse deltas
  const { panDelta, tiltDelta } = mapBoundingBoxToServoDelta(centerX, centerY);

  // Move servos relatively
 //servoSystem.moveToPositionRelative(panDelta, tiltDelta);
  
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

  // Start video stream if not already started
  if (!videoStreamStarted) {
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
      let time1 = new Date().getTime()
      await relayControllerSolenoid.activateRelayByTime(duration);
      let time2 = new Date().getTime()
      console.log(`Activating water was ${Math.floor(time2-time1)}ms...`);
      socket.emit('waterActivated', { success: true });
      console.log('Water activation completed');
    } catch (error) {
      console.error('Error activating water:', error);
      socket.emit('error', { message: 'Failed to activate water' });
    }
  });

    // Handle water activation
    socket.on('activatePump', async (duration = false) => {
        try {
            console.log(`Activating pump for ${duration || 'unlimited' }ms...`);
            if (duration) {
              await relayControllerPump.activateRelayByTime(duration);
            } else {
              await relayControllerPump.activateRelayUnlimited();
            }
            socket.emit('pumpActivated', { success: true });
            console.log('Pump activation completed');
        } catch (error) {
            console.error('Error activating pump:', error);
            socket.emit('error', { message: 'Failed to activate pump' });
        }
    });
    socket.on('deactivatePump', async () => {
      try {
          console.log(`Deactivating pump...`);
          await relayControllerPump.deactivateRelay();
          socket.emit('pumpDeactivated', { success: true });
          console.log('Pump deactivation completed');
      } catch (error) {
          console.error('Error deactivating pump:', error);
          socket.emit('error', { message: 'Failed to deactivate pump' });
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

    // Optionally stop the video stream if no clients are connected
    if (frontendNamespace.sockets.size === 0) {
      stopVideoStream();
      videoStreamStarted = false;
    }
  });
});

// Socket connection handler for AI processor
aiNamespace.on('connection', (socket) => {
  console.log('Python AI processor connected:', socket.id);

  socket.on('aiDetections', (detections) => {
    // Broadcast detections to connected frontend clients
    if (detections.length > 0) {
        let returnObjs = [];
        for (let filteredDetection of detections) {
            const FILTER_CONFIDENCE = 0.3
            const FILTER_CLASS = 'person'
            let boundingBox = filteredDetection.values[0];
            let classId = filteredDetection.classId;
            // Look up the class name from coco.txt using the classId as index
            let className = cocoLabels[classId] || `Unknown (${classId})`;
            if (FILTER_CLASS && FILTER_CLASS !== className) continue
            if (FILTER_CONFIDENCE > boundingBox[4]) continue
            returnObjs.push({
                box: boundingBox,
                meta: {
                    className,
                    classId,  // Optional: include the original classId if needed
                    confidence: boundingBox[4]
                }
            });
        }
        if (returnObjs.length > 0 && timer < (new Date().getTime() - 150)) {
            frontendNamespace.emit('detections', returnObjs);

            const [ymin, xmin, ymax, xmax] = returnObjs[0].box;
            
            // Check if the center point (0.5, 0.5) is within the bounding box
            const isCenterCovered = (
                0.5 >= xmin && 
                0.5 <= xmax && 
                0.5 >= ymin && 
                0.5 <= ymax
            );

            const FULL_SPEED = xmax > 0.95 || ymax > 0.95 || ymin > 0.95 || xmin > 0.95 ? true : false
        
            // Only move if center is not covered by the box
            if (!isCenterCovered) {
                const centerX = (xmin + xmax) / 2;
                const centerY = (ymin + ymax) / 2;
        
                // Calculate distance from center (0 to 1)
                const deltaX = (centerX - 0.5) * 2;
                const deltaY = (centerY - 0.5) * 2;
                
                // Define threshold for near/far
                const centerThreshold = 0.2;  // 20% from center
                const mediumThreshold = 0.4;  // 20% from center

                
                // Determine speed for each axis
                const xSpeed = Math.abs(deltaX) > centerThreshold ?  Math.abs(deltaX) > mediumThreshold ? 4 : 2 : 1;
                const ySpeed = Math.abs(deltaY) > centerThreshold ?  Math.abs(deltaY) > mediumThreshold ? 4 : 2 : 1;
        
                const panMovement = -Math.sign(deltaX) * (FULL_SPEED ? 10 : xSpeed);
                const tiltMovement = -Math.sign(deltaY) * (FULL_SPEED ? 10 : ySpeed);   // Negative for Y reversal
                timer = new Date().getTime()

                servoSystem.moveToPositionRelative(panMovement, tiltMovement);
            }
        
        }
    }    

    
    // Implement logic to move servos based on detections
    /*if (detections && detections.length > 0) {
      const personDetection = detections[0];  // Using the first detected person
     //box=[0.703125, 0.3421875, 0.75, 0.390625]
      adjustServosToFollow(personDetection.box);
    }*/
  });

  socket.on('disconnect', () => {
    console.log('Python AI processor disconnected:', socket.id);
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
    await relayControllerSolenoid.cleanup();
    await relayControllerPump.cleanup();
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

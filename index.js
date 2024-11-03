const http = require('http');
const socketIo = require('socket.io');
const { startServoTest, centerServos, moveToPosition } = require('./servo');
const { captureImage, removeImage } = require('./camera');
const { ServoController } = require('./relay');

let relayController = new ServoController(588); // Replace with appropriate pin number

// Create server
const server = http.createServer();
const io = socketIo(server, {
  cors: {
    origin: "*", // Replace "*" with your frontend domain in production for better security
    methods: ["GET", "POST"]
  }
});

server.listen(3000, async () => {
  console.log('Socket server running on port 3000');

  // Initialize components
  try {
    await relayController.init();
    console.log('Starting servo test...');
    await startServoTest();
    console.log('Initialization completed, waiting for commands from frontend.');
  } catch (error) {
    console.error('Initialization error:', error);
  }
});

// Handle incoming socket connections
io.on('connection', (socket) => {
  console.log('New client connected:', socket.id);

  socket.on('takePhoto', async () => {
    try {
      console.log('Taking photo...');
      const imagePath = await captureImage();
      const imageData = require('fs').readFileSync(imagePath);
      const base64Image = imageData.toString('base64');
      // Send result to the requesting client
      socket.emit('detection', {
        detected: true,
        timestamp: Date.now(),
        image: base64Image,
      });
      console.log('Photo taken and sent to frontend.');
      await removeImage();
    } catch (error) {
      console.error('Error taking photo:', error);
    }
  });

  socket.on('moveServo', async ({ pan, tilt }) => {
    try {
      console.log(`Moving servo to position: pan=${pan}, tilt=${tilt}`);
      await moveToPosition(pan, tilt);
      console.log('Servo moved to requested position.');
    } catch (error) {
      console.error('Error moving servo:', error);
    }
  });

  socket.on('centerServo', async () => {
    try {
      console.log('Centering servos...');
      await centerServos();
      console.log('Servos centered.');
    } catch (error) {
      console.error('Error centering servos:', error);
    }
  });

  socket.on('activateWater', async (duration) => {
    try {
      console.log(`Activating water for ${duration} ms...`);
      await relayController.activateWater(duration);
      console.log('Water activation completed.');
    } catch (error) {
      console.error('Error activating water:', error);
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Handle Ctrl+C
process.on('SIGINT', async () => {
  console.log('\nReceived SIGINT. Cleaning up...');
  try {
    await relayController.cleanup();
  } catch (error) {
    console.error('Error during emergency cleanup:', error);
  }
  process.exit();
});

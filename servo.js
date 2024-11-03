const i2cBus = require('i2c-bus');
const Pca9685Driver = require('pca9685').Pca9685Driver;

// Configuration options for the PCA9685 servo driver
const options = {
  i2c: i2cBus.openSync(1),
  address: 0x40, // Default I2C address for PCA9685
  frequency: 50, // Standard frequency for servos (50Hz)
  debug: false,
};

// Stores the current position and the command queue
let currentPositionServo = {
  pan: null,
  tilt: null,
  commandQue: [],
};

// Initialize the PCA9685 servo driver
const pwm = new Pca9685Driver(options, (err) => {
  if (err) {
    console.error('Error initializing PCA9685');
    process.exit(-1);
  }
  console.log('PCA9685 initialized');
  startServoPulse(); // Start processing the command queue once initialized
});

// Channels for the pan and tilt servos
const panChannel = 0;
const tiltChannel = 1;

const TILT_MAX_DOWN_PULSE = 1350;
const TILT_MAX_UP_PULSE = 2400;

const PAN_MAX_RIGHT_PULSE = 1200;
const PAN_MAX_LEFT_PULSE = 2000;

const TILT_CENTER_PULSE = Math.round((TILT_MAX_DOWN_PULSE + TILT_MAX_UP_PULSE) / 2);
const PAN_CENTER_PULSE = Math.round((PAN_MAX_RIGHT_PULSE + PAN_MAX_LEFT_PULSE) / 2);

// Function to start processing the servo pulse queue
async function startServoPulse() {
  while (true) {
    if (currentPositionServo.commandQue.length > 0) {
      let firstTask = currentPositionServo.commandQue.shift();
      try {
        // Execute the command
        await executeServoCommand(firstTask.channel, firstTask.pulse);
      } catch (error) {
        console.error('Error executing servo command:', error);
      }
    } else {
      // No commands to process, introduce a short delay to avoid busy looping
      await delay(100);
    }
  }
}

// Helper function to execute a servo command
async function executeServoCommand(channel, pulse) {
  if (channel === tiltChannel && (pulse < TILT_MAX_DOWN_PULSE || pulse > TILT_MAX_UP_PULSE)) {
    console.log('TILT: Pulse out of range');
    return;
  } else if (channel === panChannel && (pulse < PAN_MAX_RIGHT_PULSE || pulse > PAN_MAX_LEFT_PULSE)) {
    console.log('PAN: Pulse out of range');
    return;
  }

  // Update current servo position
  if (channel === panChannel) {
    currentPositionServo.pan = pulse;
  } else if (channel === tiltChannel) {
    currentPositionServo.tilt = pulse;
  }

  // Set the pulse length for the servo driver
  await pwm.setPulseLength(channel, pulse);

  // Delay to allow the servo to move to the desired position
  await delay(1200); // Adjust based on the speed of your servo
}

// Helper function to add a delay
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Function to add a command to the queue
function setServoPulse(channel, pulse) {
  currentPositionServo.commandQue.push({ channel, pulse });
}

// Initial servo test function
async function startServoTest() {
  console.log('Centering both servos...');
  setServoPulse(tiltChannel, TILT_CENTER_PULSE); // Center tilt
  setServoPulse(panChannel, PAN_CENTER_PULSE); // Center pan
  await delay(5000); // Allow enough time for servos to reach their positions

  console.log('MAX RIGHT, MAX UP');
  setServoPulse(panChannel, PAN_MAX_RIGHT_PULSE);
  setServoPulse(tiltChannel, TILT_MAX_UP_PULSE);
  await delay(1000);

  console.log('MAX LEFT, MAX DOWN');
  setServoPulse(panChannel, PAN_MAX_LEFT_PULSE);
  setServoPulse(tiltChannel, TILT_MAX_DOWN_PULSE);
  await delay(1000);

  console.log('MAX RIGHT, MAX DOWN');
  setServoPulse(panChannel, PAN_MAX_RIGHT_PULSE);
  setServoPulse(tiltChannel, TILT_MAX_DOWN_PULSE);
  await delay(1000);

  console.log('MAX LEFT, MAX UP');
  setServoPulse(panChannel, PAN_MAX_LEFT_PULSE);
  setServoPulse(tiltChannel, TILT_MAX_UP_PULSE);
  await delay(1000);

  // Randomized movements within defined ranges
  for (let i = 0; i < 5; i++) {
    const randomPanPulse = getRandomPulse(PAN_MAX_LEFT_PULSE, PAN_MAX_RIGHT_PULSE);
    const randomTiltPulse = getRandomPulse(TILT_MAX_DOWN_PULSE, TILT_MAX_UP_PULSE);

    console.log(`RANDOM MOVE ${i + 1}: PAN ${randomPanPulse}, TILT ${randomTiltPulse}`);
    setServoPulse(panChannel, randomPanPulse);
    setServoPulse(tiltChannel, randomTiltPulse);
    await delay(1000);
  }

  console.log('CENTER');
  centerServos();
  await delay(8000);

  pwm.dispose();
}

// Function to center both servos, adding tasks to the queue
function centerServos() {
  setServoPulse(panChannel, PAN_CENTER_PULSE); // Center pan
  setServoPulse(tiltChannel, TILT_CENTER_PULSE); // Center tilt
}

// Function to move servos to a specific position, adding tasks to the queue
function moveToPosition(panPulse, tiltPulse) {
  if (panPulse < PAN_MAX_RIGHT_PULSE || panPulse > PAN_MAX_LEFT_PULSE) {
    throw new Error('Pan pulse out of range');
  }
  if (tiltPulse < TILT_MAX_DOWN_PULSE || tiltPulse > TILT_MAX_UP_PULSE) {
    throw new Error('Tilt pulse out of range');
  }
  setServoPulse(panChannel, panPulse);
  setServoPulse(tiltChannel, tiltPulse);
}

// Function to move servos to a relative position, adding tasks to the queue
function moveToPositionRelative(panPulseRel, tiltPulseRel) {
  let newPan = currentPositionServo.pan + panPulseRel;
  let newTilt = currentPositionServo.tilt + tiltPulseRel;

  if (newPan < PAN_MAX_RIGHT_PULSE || newPan > PAN_MAX_LEFT_PULSE) {
    throw new Error('Pan pulse out of range');
  }
  if (newTilt < TILT_MAX_DOWN_PULSE || newTilt > TILT_MAX_UP_PULSE) {
    throw new Error('Tilt pulse out of range');
  }

  setServoPulse(panChannel, newPan);
  setServoPulse(tiltChannel, newTilt);
}

// Function to generate a random pulse within a range
function getRandomPulse(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Export the functions
module.exports = {
  startServoTest,
  centerServos,
  moveToPosition,
  moveToPositionRelative,
};

// Example usage (for testing purposes)
// (async () => {
//   try {
//     await startServoTest();
//     await centerServos();
//     await moveToPosition(1500, 1600);
//   } catch (error) {
//     console.error(error);
//   }
// })();

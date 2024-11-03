const i2cBus = require('i2c-bus')
const Pca9685Driver = require('pca9685').Pca9685Driver

// Configuration options for the PCA9685 servo driver
const options = {
  i2c: i2cBus.openSync(1),
  address: 0x40, // Default I2C address for PCA9685
  frequency: 50, // Standard frequency for servos (50Hz)
  debug: false,
}
let currentPositionServo = {
  pan: PAN_CENTER_PULSE,
  tilt: TILT_CENTER_PULSE,
  commandQue: [],
  processing: false,
};

// Function to start processing the queue
async function processQueue() {
  if (currentPositionServo.processing) {
    return; // Already processing, so return
  }

  currentPositionServo.processing = true;

  while (currentPositionServo.commandQue.length > 0) {
    const task = currentPositionServo.commandQue.shift(); // Get the first task from the queue

    try {
      await setServoPulse(task.channel, task.pulse);
    } catch (error) {
      console.error('Error executing servo command:', error);
    }
  }

  currentPositionServo.processing = false;
}

// Helper function to set servo pulse length and ensure it finishes before continuing
async function setServoPulse(channel, pulse) {
  if (channel === tiltChannel && pulse < TILT_MAX_DOWN_PULSE) {
    console.log('TILT: Max. DOWN reached');
    return false;
  } else if (channel === tiltChannel && pulse > TILT_MAX_UP_PULSE) {
    console.log('TILT: Max. UP reached');
    return false;
  } else if (channel === panChannel && pulse > PAN_MAX_LEFT_PULSE) {
    console.log('PAN: Max. LEFT reached');
    return false;
  } else if (channel === panChannel && pulse < PAN_MAX_RIGHT_PULSE) {
    console.log('PAN: Max. RIGHT reached');
    return false;
  }

  // Update current position
  if (channel === panChannel) {
    currentPositionServo.pan = pulse;
  } else if (channel === tiltChannel) {
    currentPositionServo.tilt = pulse;
  }

  // Set the pulse length for the servo driver
  await pwm.setPulseLength(channel, pulse);
  await delay(500); // Add a delay to give time for the servo to reach the position
}

// Function to start the servo test, adding tasks to the queue
async function startServoTest() {
  try {
    console.log('Centering both servos...');
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_CENTER_PULSE }); // Center tilt
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_CENTER_PULSE }); // Center pan

    console.log('MAX RIGHT, MAX UP');
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_MAX_RIGHT_PULSE });
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_MAX_UP_PULSE });

    console.log('MAX LEFT, MAX DOWN');
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_MAX_LEFT_PULSE });
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_MAX_DOWN_PULSE });

    console.log('MAX RIGHT, MAX DOWN');
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_MAX_RIGHT_PULSE });
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_MAX_DOWN_PULSE });

    console.log('MAX LEFT, MAX UP');
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_MAX_LEFT_PULSE });
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_MAX_UP_PULSE });

    // Randomized movements within defined ranges
    for (let i = 0; i < 5; i++) {
      const randomPanPulse = getRandomPulse(PAN_MAX_LEFT_PULSE, PAN_MAX_RIGHT_PULSE);
      const randomTiltPulse = getRandomPulse(TILT_MAX_DOWN_PULSE, TILT_MAX_UP_PULSE);

      console.log(`RANDOM MOVE ${i + 1}: PAN ${randomPanPulse}, TILT ${randomTiltPulse}`);
      currentPositionServo.commandQue.push({ channel: panChannel, pulse: randomPanPulse });
      currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: randomTiltPulse });
    }

    console.log('CENTER');
    currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_CENTER_PULSE });
    currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_CENTER_PULSE });

    // Start processing the queue
    processQueue();
  } catch (error) {
    console.error('Error during servo test:', error);
    pwm.dispose();
    process.exit(1);
  }
}

// Function to center both servos, adding tasks to the queue
async function centerServos() {
  currentPositionServo.commandQue.push({ channel: panChannel, pulse: PAN_CENTER_PULSE });
  currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: TILT_CENTER_PULSE });
  processQueue();
}

// Function to move servos to a specific position, adding tasks to the queue
async function moveToPosition(panPulse, tiltPulse) {
  if (panPulse < PAN_MAX_RIGHT_PULSE || panPulse > PAN_MAX_LEFT_PULSE) {
    throw new Error('Pan pulse out of range');
  }
  if (tiltPulse < TILT_MAX_DOWN_PULSE || tiltPulse > TILT_MAX_UP_PULSE) {
    throw new Error('Tilt pulse out of range');
  }
  currentPositionServo.commandQue.push({ channel: panChannel, pulse: panPulse });
  currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: tiltPulse });
  processQueue();
}

// Function to move servos to a relative position, adding tasks to the queue
async function moveToPositionRelative(panPulseRel, tiltPulseRel) {
  let newPan = currentPositionServo.pan + panPulseRel;
  let newTilt = currentPositionServo.tilt + tiltPulseRel;

  if (newPan < PAN_MAX_RIGHT_PULSE || newPan > PAN_MAX_LEFT_PULSE) {
    throw new Error('Pan pulse out of range');
  }
  if (newTilt < TILT_MAX_DOWN_PULSE || newTilt > TILT_MAX_UP_PULSE) {
    throw new Error('Tilt pulse out of range');
  }

  currentPositionServo.commandQue.push({ channel: panChannel, pulse: newPan });
  currentPositionServo.commandQue.push({ channel: tiltChannel, pulse: newTilt });
  processQueue();
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
  moveToPositionRelative
};

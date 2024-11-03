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
  pan: null, 
  tilt: null, 
  commandQue: []
}

// Initialize the PCA9685 servo driver
const pwm = new Pca9685Driver(options, (err) => {
  if (err) {
    console.error('Error initializing PCA9685')
    process.exit(-1)
  }

  console.log('PCA9685 initialized')
})

// Channels for the pan and tilt servos
const panChannel = 0
const tiltChannel = 1

const TILT_MAX_DOWN_PULSE = 1350
const TILT_MAX_UP_PULSE = 2400

const PAN_MAX_RIGHT_PULSE = 1200
const PAN_MAX_LEFT_PULSE = 2000

const TILT_CENTER_PULSE = Math.round((TILT_MAX_DOWN_PULSE + TILT_MAX_UP_PULSE) / 2)
const PAN_CENTER_PULSE = Math.round((PAN_MAX_RIGHT_PULSE + PAN_MAX_LEFT_PULSE) / 2)

async function startServoPulse() {
  while (true) {
    if (currentPositionServo.commandQue.length > 0) {
      let firstTask = currentPositionServo.commandQue.shift();
      if (firstTask.channel === 1 && firstTask.pulse < TILT_MAX_DOWN_PULSE) {
        console.log('TILT: Max. DOWN reached')
        return false
      } else if (firstTask.channel === 1 && firstTask.pulse > TILT_MAX_UP_PULSE) {
        console.log('TILT: Max. UP reached')
        return false
      } else if (firstTask.channel === 0 && firstTask.pulse > PAN_MAX_LEFT_PULSE) {
        console.log('PAN: Max. LEFT reached')
        return false
      } else if (firstTask.channel === 0 && firstTask.pulse < PAN_MAX_RIGHT_PULSE) {
        console.log('PAN: Max. RIGHT reached')
        return false
      }
    
      // Update current servo position
      if (firstTask.channel === 0) {
        currentPositionServo.pan = firstTask.pulse
      } else {
        currentPositionServo.tilt = firstTask.pulse
      }
    
      // Set the pulse length for the servo driver
      pwm.setPulseLength(firstTask.channel, firstTask.pulse)
    
      // Introduce delay to allow servo to reach the desired position
      // You may need to adjust this delay to fit the specific servo speed
      await delay(1200) // 500ms delay is arbitrary, adjust based on your servo's speed    
    }
  }
}


// Helper function to add a delay
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Function to start the servo test, adding tasks to the queue
async function startServoTest() {
  try {
    startServoPulse()
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
}


// Function to generate a random pulse within a range
function getRandomPulse(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

// Export the functions
module.exports = {
  startServoTest,
  centerServos,
  moveToPosition,
  moveToPositionRelative
}

// Example usage (for testing purposes)
// (async () => {
//   try {
//     await startServoTest()
//     await centerServos()
//     await moveToPosition(1500, 1600)
//   } catch (error) {
//     console.error(error)
//   }
// })()

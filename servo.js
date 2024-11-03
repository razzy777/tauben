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
  tilt: null
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

// Helper function to set servo pulse length
function setServoPulse(channel, pulse) {
  if (channel === 1 && pulse < TILT_MAX_DOWN_PULSE) {
    console.log('TILT: Max. DOWN reached')
    return false
  } else if (channel === 1 && pulse > TILT_MAX_UP_PULSE) {
    console.log('TILT: Max. UP reached')
    return false
  } else if (channel === 0 && pulse > PAN_MAX_LEFT_PULSE) {
    console.log('PAN: Max. LEFT reached')
    return false
  } else if (channel === 0 && pulse < PAN_MAX_RIGHT_PULSE) {
    console.log('PAN: Max. RIGHT reached')
    return false
  }
  if (channel === 0) {
    currentPositionServo.pan = pulse
  } else  {
    currentPositionServo.tilt = pulse
  }
  pwm.setPulseLength(channel, pulse)
}

// Helper function to add a delay
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Initial servo test function
async function startServoTest() {
  try {
    console.log('Centering both servos...')
    setServoPulse(tiltChannel, TILT_CENTER_PULSE) // Center tilt
    await delay(5000)

    console.log('MAX RIGHT, MAX UP')
    setServoPulse(panChannel, PAN_MAX_RIGHT_PULSE) // Move pan right
    setServoPulse(tiltChannel, TILT_MAX_UP_PULSE) // Move tilt up
    await delay(1000)

    console.log('MAX LEFT, MAX DOWN')
    setServoPulse(panChannel, PAN_MAX_LEFT_PULSE) // Move pan left
    setServoPulse(tiltChannel, TILT_MAX_DOWN_PULSE) // Move tilt down
    await delay(1000)

    console.log('MAX RIGHT, MAX DOWN')
    setServoPulse(panChannel, PAN_MAX_RIGHT_PULSE) // Move pan right
    setServoPulse(tiltChannel, TILT_MAX_DOWN_PULSE) // Move tilt down
    await delay(1000)

    console.log('MAX LEFT, MAX UP')
    setServoPulse(panChannel, PAN_MAX_LEFT_PULSE) // Move pan left
    setServoPulse(tiltChannel, TILT_MAX_UP_PULSE) // Move tilt up
    await delay(1000)

    // Randomized movements within defined ranges
    for (let i = 0; i < 5; i++) {
      const randomPanPulse = getRandomPulse(PAN_MAX_LEFT_PULSE, PAN_MAX_RIGHT_PULSE)
      const randomTiltPulse = getRandomPulse(TILT_MAX_DOWN_PULSE, TILT_MAX_UP_PULSE)

      console.log(`RANDOM MOVE ${i + 1}: PAN ${randomPanPulse}, TILT ${randomTiltPulse}`)
      setServoPulse(panChannel, randomPanPulse)
      setServoPulse(tiltChannel, randomTiltPulse)

      await delay(1000)
    }

    console.log('CENTER')
    await centerServos()
    await delay(8000)

    pwm.dispose()
  } catch (error) {
    console.error('Error during servo test:', error)
    pwm.dispose()
    process.exit(1)
  }
}

// Function to center both servos
async function centerServos() {
  setServoPulse(panChannel, PAN_CENTER_PULSE) // Center pan
  setServoPulse(tiltChannel, TILT_CENTER_PULSE) // Center tilt
}

// Function to move servos to a specific position
async function moveToPosition(panPulse, tiltPulse) {
  if (panPulse < PAN_MAX_RIGHT_PULSE || panPulse > PAN_MAX_LEFT_PULSE) {
    throw new Error('Pan pulse out of range')
  }
  if (tiltPulse < TILT_MAX_DOWN_PULSE || tiltPulse > TILT_MAX_UP_PULSE) {
    throw new Error('Tilt pulse out of range')
  }
  setServoPulse(panChannel, panPulse)
  setServoPulse(tiltChannel, tiltPulse)
}

// Function to move servos to a specific position
async function moveToPositionRelative(panPulseRel, tiltPulseRel) {
  let newPan = panPulseRel ? currentPositionServo.pan + panPulseRel : null
  let newTilt = tiltPulseRel ? currentPositionServo.tile + tiltPulseRel : null
  if (newPan && (newPan < PAN_MAX_RIGHT_PULSE || newPan > PAN_MAX_LEFT_PULSE)) {
    throw new Error('Pan pulse out of range')
  } else if (newPan) {
    setServoPulse(panChannel, newPan)
  }
  if (newTilt && (newTilt < TILT_MAX_DOWN_PULSE || newTilt > TILT_MAX_UP_PULSE)) {
    throw new Error('Tilt pulse out of range')
  } else if (newTilt) {
    setServoPulse(tiltChannel, newTilt)
  }
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

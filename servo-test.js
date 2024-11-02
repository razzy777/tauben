const i2cBus = require('i2c-bus')
const Pca9685Driver = require('pca9685').Pca9685Driver

// Configuration options for the PCA9685 servo driver
const options = {
  i2c: i2cBus.openSync(1),
  address: 0x40, // Default I2C address for PCA9685
  frequency: 50, // Standard frequency for servos (50Hz)
  debug: false,
}

// Initialize the PCA9685 servo driver
const pwm = new Pca9685Driver(options, (err) => {
  if (err) {
    console.error('Error initializing PCA9685')
    process.exit(-1)
  }

  console.log('PCA9685 initialized')
  startServoTest()
})

// Channels for the pan and tilt servos
const panChannel = 0
const tiltChannel = 1

const TILT_MAX_DOWN_PULSE = 1350
const TILT_MAX_UP_PULSE = 2500

const PAN_MAX_RIGHT_PULSE = 1400
const PAN_MAX_LEFT_PULSE = 1600

const TILT_CENTER_PULSE = Math.round((TILT_MAX_DOWN_PULSE + TILT_MAX_UP_PULSE)/2)
const PAN_CENTER_PULSE = Math.round((PAN_MAX_RIGHT_PULSE + PAN_MAX_LEFT_PULSE)/2)


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
    pwm.setPulseLength(channel, pulse)
}

// Helper function to add a delay
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Main function to execute the test movements
async function startServoTest() {
  try {
    console.log('Centering both servos...')
    //setServoPulse(panChannel, 1500) // Center pan
    setServoPulse(tiltChannel, TILT_CENTER_PULSE) // Center tilt
    await delay(5000)
    // 1. Move Pan Servo Left and Right
    //console.log('Testing pan servo: left, right, center')
    //setServoPulse(panChannel, 1000) // Move pan servo left
    //await delay(1000)
    ///setServoPulse(panChannel, 2000) // Move pan servo right
    //await delay(1000)
    //setServoPulse(panChannel, 1500) // Return pan to center
    //await delay(1000)

    // 2. Move Tilt Servo Up and Down
    console.log('MAX RIGHT')
    setServoPulse(panChannel, PAN_MAX_RIGHT_PULSE) // Move tilt servo up
    await delay(1000)
    console.log('MAX LEFT')
    setServoPulse(panChannel, PAN_MAX_LEFT_PULSE) // Move tilt servo down
    await delay(1000)
    // setServoPulse(tiltChannel, 1600) // Return tilt to center
    // await delay(1000)

    // 3. Pan Servo Sweeping Left to Right
    /*console.log('Sweeping pan servo from left to right')
    for (let pulse = 1000; pulse <= 2000; pulse += 250) {
      setServoPulse(panChannel, pulse)
      await delay(500) // Pause to observe each position
    }
    setServoPulse(panChannel, 1500) // Return to center
    await delay(1000)

    // 4. Diagonal Movement: Bottom-Left to Top-Right
    console.log('Diagonal movement: bottom-left to top-right')
    setServoPulse(panChannel, 1000) // Move pan to left
    setServoPulse(tiltChannel, 2000) // Move tilt down
    await delay(1000)

    setServoPulse(panChannel, 2000) // Move pan to right
    setServoPulse(tiltChannel, 1000) // Move tilt up
    await delay(1000)
    
    setServoPulse(panChannel, 1500) // Center pan
    setServoPulse(tiltChannel, TILT_CENTER_PULSE) // Center tilt
    await delay(1000)

    console.log('Test movements completed.')
    */
    // Cleanup: Turn off PWM output to stop any servo signals
    setServoPulse(panChannel, PAN_CENTER_PULSE) // Center pan
    setServoPulse(tiltChannel, TILT_CENTER_PULSE) // Center tilt

    pwm.dispose()
    process.exit(0)
  } catch (error) {
    console.error('Error during servo test:', error)
    pwm.dispose()
    process.exit(1)
  }
}

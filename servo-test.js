const i2cBus = require('i2c-bus');
const Pca9685Driver = require('pca9685').Pca9685Driver;

// Configuration options for the PCA9685 servo driver
const options = {
    i2c: null,          // We'll initialize this after checking the bus
    address: 0x40,      // Default I2C address for PCA9685
    frequency: 50,      // Standard frequency for servos (50Hz)
    debug: false,
};

// Constants
const TILT_MAX_DOWN_PULSE = 1350;
const TILT_MAX_UP_PULSE = 2500;
const TILT_CENTER_PULSE = Math.round((TILT_MAX_DOWN_PULSE + TILT_MAX_UP_PULSE)/2);
const panChannel = 0;
const tiltChannel = 1;

// Helper function to set servo pulse length with error checking
function setServoPulse(channel, pulse) {
    try {
        if (channel === 1) {
            if (pulse < TILT_MAX_DOWN_PULSE) {
                console.log('TILT: Max. DOWN reached');
                return false;
            } else if (pulse > TILT_MAX_UP_PULSE) {
                console.log('TILT: Max. UP reached');
                return false;
            }
        }
        pwm.setPulseLength(channel, pulse);
        return true;
    } catch (error) {
        console.error(`Error setting pulse for channel ${channel}:`, error);
        return false;
    }
}

// Helper function to add a delay
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Initialize I2C and PCA9685
async function initializeHardware() {
    try {
        // First check if I2C bus is available
        const i2c = await i2cBus.openPromisified(1);
        
        // Scan for the PCA9685 device
        try {
            await i2c.scan();
            console.log('I2C bus scanned successfully');
        } catch (error) {
            console.error('Error scanning I2C bus:', error);
            throw error;
        }

        // Now initialize the driver
        options.i2c = i2cBus.openSync(1);
        
        return new Promise((resolve, reject) => {
            const pwm = new Pca9685Driver(options, (err) => {
                if (err) {
                    console.error('Error initializing PCA9685:', err);
                    reject(err);
                    return;
                }
                console.log('PCA9685 initialized successfully');
                resolve(pwm);
            });
        });
    } catch (error) {
        console.error('Error during hardware initialization:', error);
        throw error;
    }
}

// Main function to execute the test movements
async function startServoTest(pwm) {
    try {
        console.log('Centering both servos...');
        setServoPulse(tiltChannel, TILT_CENTER_PULSE);
        await delay(1000);  // Reduced delay for testing

        console.log('Testing tilt servo...');
        if (setServoPulse(panChannel, 500)) {
            await delay(1000);
        }
        
        if (setServoPulse(panChannel, 2500)) {
            await delay(1000);
        }
        
        if (setServoPulse(tiltChannel, 1600)) {
            await delay(1000);
        }

        console.log('Returning to center position...');
        setServoPulse(panChannel, 1500);
        setServoPulse(tiltChannel, 1500);
        await delay(1000);

        console.log('Test movements completed.');
    } catch (error) {
        console.error('Error during servo test:', error);
    } finally {
        try {
            if (pwm) {
                pwm.dispose();
            }
        } catch (error) {
            console.error('Error disposing PWM:', error);
        }
    }
}

// Main execution
async function main() {
    let pwm = null;
    try {
        // Check I2C permissions
        if (process.getuid() !== 0) {
            console.error('This script must be run with sudo privileges');
            process.exit(1);
        }

        pwm = await initializeHardware();
        await startServoTest(pwm);
    } catch (error) {
        console.error('Fatal error:', error);
    } finally {
        try {
            if (pwm) {
                pwm.dispose();
            }
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
        process.exit(0);
    }
}

// Handle Ctrl+C and other termination signals
process.on('SIGINT', () => {
    console.log('\nReceived SIGINT. Cleaning up...');
    if (pwm) {
        try {
            pwm.dispose();
        } catch (error) {
            console.error('Error during emergency cleanup:', error);
        }
    }
    process.exit(0);
});

// Start the program
main().catch(console.error);
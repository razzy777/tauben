const Gpio = require('onoff').Gpio;

class ServoController {
    constructor(pin) {
        this.pin = pin;
        this.servo = null;
    }

    async init() {
        try {
            console.log(`Initializing solenoid on pin ${this.pin}...`);
            this.servo = new Gpio(this.pin, 'out');
            console.log('solenoid initialized');
        } catch (error) {
            console.error('Error initializing solenoid:', error);
            throw error;
        }
    }

    async activate() {
        try {
            console.log('Activating solenoid...');
            // First unexport
            this.servo.unexport();
            await this.delay(100);
            
            // Reinitialize and set high
            this.servo = new Gpio(this.pin, 'out');
            await this.servo.write(1);
            console.log('Relay activated');
        } catch (error) {
            console.error('Error activating solenoid:', error);
            throw error;
        }
    }

    async deactivate() {
        try {
            console.log('Deactivating Relay...');
            // First unexport
            this.servo.unexport();
            await this.delay(100);
            
            // Reinitialize and set low
            this.servo = new Gpio(this.pin, 'out');
            await this.servo.write(0);
            console.log('Relay deactivated');
        } catch (error) {
            console.error('Error deactivating Relay:', error);
            throw error;
        }
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async cleanup() {
        try {
            if (this.servo) {
                await this.deactivate();
                this.servo.unexport();
                console.log('solenoid cleanup completed');
            }
        } catch (error) {
            console.error('Error during cleanup:', error);
            throw error;
        }
    }

    // Function to activate the relay for a specific time in milliseconds
    async activateWater(duration) {
        try {
            await this.activate();
            await this.delay(duration);
            await this.deactivate();
            console.log(`Relay activated for ${duration}ms`);
        } catch (error) {
            console.error('Error during activateWater:', error);
            throw error;
        }
    }
}

// Export the functions
module.exports = {
    ServoController,
}

// Example usage (for testing purposes)
// (async () => {
//     const servo = new ServoController(588);
//     try {
//         await servo.init();
//         await servo.activateWater(500);
//     } catch (error) {
//         console.error(error);
//     } finally {
//         await servo.cleanup();
//     }
// })();

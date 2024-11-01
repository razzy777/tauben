const Gpio = require('onoff').Gpio;

class ServoController {
    constructor(pin) {
        this.pin = pin;
        this.servo = null;
    }

    async init() {
        try {
            console.log(`Initializing servo on pin ${this.pin}...`);
            this.servo = new Gpio(this.pin, 'out');
            console.log('Servo initialized');
        } catch (error) {
            console.error('Error initializing servo:', error);
            throw error;
        }
    }

    async activate() {
        try {
            console.log('Activating servo...');
            await this.servo.write(1);
            console.log('Servo activated');
        } catch (error) {
            console.error('Error activating servo:', error);
            throw error;
        }
    }

    async deactivate() {
        try {
            console.log('Deactivating servo...');
            await this.servo.write(0);
            console.log('Servo deactivated');
        } catch (error) {
            console.error('Error deactivating servo:', error);
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
                console.log('Servo cleanup completed');
            }
        } catch (error) {
            console.error('Error during cleanup:', error);
            throw error;
        }
    }
}

// Example usage functions
async function testServo() {
    const servo = new ServoController(588);
    
    try {
        await servo.init();
        await servo.activate();
        await servo.delay(1000);
        await servo.deactivate();
    } catch (error) {
        console.error('Error during servo test:', error);
    } finally {
        await servo.cleanup();
    }
}

// More complex movement pattern example
async function servoPattern() {
    const servo = new ServoController(588);
    
    try {
        await servo.init();
        
        // Pattern: on-off-on-off
        for (let i = 0; i < 2; i++) {
            await servo.activate();
            await servo.delay(500);
            await servo.deactivate();
            await servo.delay(500);
        }
    } catch (error) {
        console.error('Error during servo pattern:', error);
    } finally {
        await servo.cleanup();
    }
}

// Handle Ctrl+C
process.on('SIGINT', async () => {
    console.log('\nReceived SIGINT. Cleaning up...');
    try {
        if (servo) {
            await servo.cleanup();
        }
    } catch (error) {
        console.error('Error during emergency cleanup:', error);
    }
    process.exit();
});

// You can now use either of these functions:
// testServo().catch(console.error);
// servoPattern().catch(console.error);

// Or create your own custom patterns:
async function customPattern() {
    const servo = new ServoController(588);
    
    try {
        await servo.init();
        
        // Your custom pattern here
        await servo.activate();
        await servo.delay(200);
        await servo.deactivate();
        await servo.delay(800);
        await servo.activate();
        await servo.delay(1000);
        await servo.deactivate();
    } catch (error) {
        console.error('Error during custom pattern:', error);
    } finally {
        await servo.cleanup();
    }
}

// Run your preferred pattern
customPattern().catch(console.error);
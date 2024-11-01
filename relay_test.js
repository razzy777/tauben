const Gpio = require('onoff').Gpio;

class ServoController {
    constructor(pin) {
        this.pin = pin;
        this.servo = null;
    }

    async init() {
        try {
            console.log(`Initializing servo on pin ${this.pin}...`);
            // Force cleanup of any existing GPIO setup
            try {
                const cleanup = new Gpio(this.pin, 'out');
                cleanup.unexport();
                await this.delay(100); // Wait for cleanup
            } catch (e) {
                // Ignore cleanup errors
            }
            
            this.servo = new Gpio(this.pin, 'out');
            // Ensure we start in a known state
            await this.servo.write(0);
            console.log('Servo initialized in OFF state');
        } catch (error) {
            console.error('Error initializing servo:', error);
            throw error;
        }
    }

    async activate() {
        try {
            console.log('Activating servo...');
            await this.servo.write(1);
            const state = await this.servo.read();
            console.log('Servo state after activation:', state);
        } catch (error) {
            console.error('Error activating servo:', error);
            throw error;
        }
    }

    async deactivate() {
        try {
            console.log('Deactivating servo...');
            await this.servo.write(0);
            const state = await this.servo.read();
            console.log('Servo state after deactivation:', state);
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
                await this.delay(100); // Wait before unexporting
                this.servo.unexport();
                console.log('Servo cleanup completed');
            }
        } catch (error) {
            console.error('Error during cleanup:', error);
            throw error;
        }
    }

    // Add a method to check current state
    async getState() {
        try {
            const state = await this.servo.read();
            console.log('Current servo state:', state);
            return state;
        } catch (error) {
            console.error('Error reading servo state:', error);
            throw error;
        }
    }
}

async function testServoMovement() {
    const servo = new ServoController(588);
    
    try {
        await servo.init();
        
        for (let i = 0; i < 5; i++) {
            console.log(`\nMovement cycle ${i + 1}`);
            
            // Activate
            await servo.activate();
            await servo.getState();
            await servo.delay(1000);
            
            // Deactivate
            await servo.deactivate();
            await servo.getState();
            await servo.delay(1000);
            
            // Add a small delay between cycles
            await servo.delay(500);
        }
    } catch (error) {
        console.error('Error during servo test:', error);
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

// Run the test
console.log('Starting servo movement test...');
testServoMovement().catch(console.error);
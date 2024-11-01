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
            // First unexport
            this.servo.unexport();
            await this.delay(100);
            
            // Reinitialize and set high
            this.servo = new Gpio(this.pin, 'out');
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
            // First unexport
            this.servo.unexport();
            await this.delay(100);
            
            // Reinitialize and set low
            this.servo = new Gpio(this.pin, 'out');
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

async function testServoMovement() {
    const servo = new ServoController(588);
    
    try {
        await servo.init();
        
        for (let i = 0; i < 5; i++) {
            console.log(`\nMovement cycle ${i + 1}`);
            
            await servo.activate();
            await servo.delay(1000);
            
            await servo.deactivate();
            await servo.delay(1000);
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
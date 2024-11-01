const Gpio = require('pigpio').Gpio;

try {
    console.log('Initializing GPIO 17...');
    // Configure GPIO17 as an output
    const relay = new Gpio(17, {mode: Gpio.OUTPUT});
    
    console.log('Activating relay...');
    relay.digitalWrite(1); // Turn on
    
    setTimeout(() => {
        try {
            console.log('Deactivating relay...');
            relay.digitalWrite(0); // Turn off
            console.log('Relay test completed');
        } catch (timeoutError) {
            console.error('Error in timeout callback:', timeoutError);
        }
    }, 3000);
} catch (error) {
    console.error('Error with GPIO:', error);
}
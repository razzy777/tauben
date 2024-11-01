const rpio = require('rpio');

try {
    console.log('Initializing GPIO 17...');
    
    // Configure pin 17 (physical pin 11) as output
    rpio.open(17, rpio.OUTPUT, rpio.LOW);
    
    console.log('Activating relay...');
    rpio.write(17, rpio.HIGH);
    
    setTimeout(() => {
        try {
            console.log('Deactivating relay...');
            rpio.write(17, rpio.LOW);
            rpio.close(17);
            console.log('Relay test completed');
        } catch (timeoutError) {
            console.error('Error in timeout callback:', timeoutError);
            rpio.close(17);
        }
    }, 3000);
} catch (error) {
    console.error('Error with GPIO:', error);
    try {
        rpio.close(17);
    } catch (closeError) {
        // Ignore close errors
    }
}

// Handle process termination
process.on('SIGINT', () => {
    try {
        rpio.close(17);
    } catch (error) {
        // Ignore close errors
    }
    process.exit();
});
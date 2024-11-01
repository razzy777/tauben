const Gpio = require('onoff').Gpio;

try {
    console.log('Initializing GPIO...');
    // Try using the new chip-relative numbering
    // You'll need to replace XXX with the correct offset based on gpioinfo output
    const relay = new Gpio('gpio-588', 'out');
    // Alternative format that might work:
    // const relay = new Gpio('/dev/gpiochip512', 'out');
    
    console.log('Activating relay...');
    relay.writeSync(1); // Activate relay
    
    setTimeout(() => {
        try {
            console.log('Deactivating relay...');
            relay.writeSync(0);
            relay.unexport();
            console.log('Relay test completed');
        } catch (timeoutError) {
            console.error('Error in timeout callback:', timeoutError);
            if (relay) relay.unexport();
        }
    }, 3000);
} catch (error) {
    console.error('Error with GPIO:', error);
}
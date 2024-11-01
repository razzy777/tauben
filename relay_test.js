const Gpio = require('onoff').Gpio;

try {
    console.log('Initializing GPIO 17...');
    const relay = new Gpio(17, 'out');
    
    console.log('Activating relay...');
    relay.writeSync(1); // Activate relay
    
    setTimeout(() => {
        try {
            console.log('Deactivating relay...');
            relay.writeSync(0); // Deactivate relay after 3 seconds
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
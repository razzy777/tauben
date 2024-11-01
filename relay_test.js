const Gpio = require('pigpio').Gpio;

// Initialize GPIO pin (replace 17 with your actual GPIO pin number)
const relay = new Gpio('GPIO17', { mode: Gpio.OUTPUT });

console.log('Activating relay...');
relay.digitalWrite(1); // Activate relay

setTimeout(() => {
  console.log('Deactivating relay...');
  relay.digitalWrite(0); // Deactivate relay
  relay.close(); // Release resources
  console.log('Relay test completed');
}, 3000);

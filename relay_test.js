const Gpio = require('pigpio').Gpio;

const relay = new Gpio('GPIO17', { mode: Gpio.OUTPUT }); // Use the correct GPIO pin number

// Activate the relay (turn ON solenoid valve)
relay.digitalWrite(1);

// Wait for 5 seconds
setTimeout(() => {
  // Deactivate the relay (turn OFF solenoid valve)
  relay.digitalWrite(0);
  process.exit(0); // Exit the script
}, 5000);

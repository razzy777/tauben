const Gpio = require('onoff').Gpio;

const relay = new Gpio(17, 'out'); // GPIO 17 connected to IN1

// Activate the relay (turn ON solenoid valve)
relay.writeSync(1);

// Wait for 5 seconds
setTimeout(() => {
  // Deactivate the relay (turn OFF solenoid valve)
  relay.writeSync(0);
  relay.unexport();
}, 5000);

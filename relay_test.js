const Gpio = require('onoff').Gpio;

// Create a new Gpio object with chip and line numbers
const options = {
  chip: 0,      // GPIO chip number, usually 0
  line: 17      // GPIO line number (BCM numbering)
};

const servo = new Gpio(588, 'out');

// Test by toggling the servo control line
servo.writeSync(1);
setTimeout(() => {
  servo.writeSync(0);
  servo.unexport();
  console.log('Servo test completed');
}, 1000);

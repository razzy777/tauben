const i2cBus = require('i2c-bus');
const Pca9685Driver = require('pca9685').Pca9685Driver;

const options = {
  i2c: i2cBus.openSync(1),
  address: 0x40,
  frequency: 50,
  debug: false
};

const pwm = new Pca9685Driver(options, (err) => {
  if (err) {
    console.error('Error initializing PCA9685');
    process.exit(-1);
  }

  console.log('PCA9685 initialized');

  // Servo channels
  const panChannel = 0;
  const tiltChannel = 1;

  // Function to set servo pulse length
  function setServoPulse(channel, pulse) {
    pwm.setPulseLength(channel, pulse);
  }

  // Center position (adjust pulse lengths as needed)
  setServoPulse(panChannel, 1500);
  setServoPulse(tiltChannel, 1500);

  // Add your test movements here

  // Cleanup
  setTimeout(() => {
    pwm.dispose();
    process.exit(0);
  }, 5000);
});

const i2cBus = require('i2c-bus');
const Pca9685Driver = require('pca9685').Pca9685Driver;

class ServoSystem {
  constructor() {
    this.options = {
      i2c: i2cBus.openSync(1),
      address: 0x40,
      frequency: 50,
      debug: false,
    };

    this.TILT_MAX_DOWN_PULSE = 1350;
    this.TILT_MAX_UP_PULSE = 2400;
    this.PAN_MAX_RIGHT_PULSE = 1200;
    this.PAN_MAX_LEFT_PULSE = 2000;

    this.TILT_CENTER_PULSE = Math.round((this.TILT_MAX_DOWN_PULSE + this.TILT_MAX_UP_PULSE) / 2);
    this.PAN_CENTER_PULSE = Math.round((this.PAN_MAX_RIGHT_PULSE + this.PAN_MAX_LEFT_PULSE) / 2);

    this.panChannel = 0;
    this.tiltChannel = 1;

    this.currentPosition = {
      pan: this.PAN_CENTER_PULSE,
      tilt: this.TILT_CENTER_PULSE,
    };

    this.isProcessing = {
      pan: false,
      tilt: false,
    };

    this.queues = {
      pan: [],
      tilt: [],
    };

    this.pwm = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    return new Promise((resolve, reject) => {
      this.pwm = new Pca9685Driver(this.options, (err) => {
        if (err) {
          reject(err);
          return;
        }
        this.initialized = true;
        this.startQueueProcessing();
        resolve();
      });
    });
  }

  async reinitialize() {
    if (this.pwm) {
      try {
        await this.cleanup();
      } catch (error) {
        console.error('Error during cleanup:', error);
      }
    }
    this.initialized = false;
    await this.initialize();
  }

  startQueueProcessing() {
    this.processQueue('pan');
    this.processQueue('tilt');
  }

  async processQueue(type) {
    if (this.isProcessing[type]) return;
    this.isProcessing[type] = true;

    while (true) {
      try {
        if (!this.initialized) {
          await this.reinitialize();
        }

        if (this.queues[type].length > 0) {
          const task = this.queues[type].shift();
          await this.executeServoCommand(
            type === 'pan' ? this.panChannel : this.tiltChannel,
            task.pulse
          );
        }
        await this.delay(100);
      } catch (error) {
        console.error(`Error processing ${type} queue:`, error);
        await this.delay(1000); // Wait before retrying
        try {
          await this.reinitialize();
        } catch (reinitError) {
          console.error('Error reinitializing:', reinitError);
        }
      }
    }
  }

  async executeServoCommand(channel, pulse) {
    if (!this.initialized || !this.pwm) {
      throw new Error('Servo system not initialized');
    }

    // Validate pulse ranges
    if (channel === this.tiltChannel) {
      if (pulse < this.TILT_MAX_DOWN_PULSE || pulse > this.TILT_MAX_UP_PULSE) {
        console.log('TILT: Pulse out of range');
        return;
      }
      this.currentPosition.tilt = pulse;
    } else if (channel === this.panChannel) {
      if (pulse < this.PAN_MAX_RIGHT_PULSE || pulse > this.PAN_MAX_LEFT_PULSE) {
        console.log('PAN: Pulse out of range');
        return;
      }
      this.currentPosition.pan = pulse;
    }

    try {
      await this.pwm.setPulseLength(channel, pulse);
      await this.delay(1200);
    } catch (error) {
      console.error('Error setting pulse length:', error);
      throw error; // Propagate error for reinitialize handling
    }
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async cleanup() {
    if (this.pwm) {
      try {
        await this.pwm.dispose();
        this.pwm = null;
        this.initialized = false;
      } catch (error) {
        console.error('Error disposing PWM:', error);
        throw error;
      }
    }
  }

  // Public movement methods
  moveToPosition(panPulse, tiltPulse) {
    if (panPulse) {
      this.queues.pan.push({ pulse: panPulse });
    }
    if (tiltPulse) {
      this.queues.tilt.push({ pulse: tiltPulse });
    }
  }

  moveToPositionRelative(panPulseRel, tiltPulseRel) {
    const newPan = this.currentPosition.pan + (panPulseRel || 0);
    const newTilt = this.currentPosition.tilt + (tiltPulseRel || 0);

    if (panPulseRel) {
      this.queues.pan.push({ pulse: newPan });
    }
    if (tiltPulseRel) {
      this.queues.tilt.push({ pulse: newTilt });
    }
  }

  centerServos() {
    this.moveToPosition(this.PAN_CENTER_PULSE, this.TILT_CENTER_PULSE);
  }
}

// Export a singleton instance
const servoSystem = new ServoSystem();
module.exports = servoSystem;
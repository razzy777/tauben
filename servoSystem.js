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

    // Servo configuration
    this.TILT_MAX_DOWN_PULSE = 1450;
    this.TILT_MAX_UP_PULSE = 2400;
    this.PAN_MAX_RIGHT_PULSE = 1100;
    this.PAN_MAX_LEFT_PULSE = 2400;

    this.TILT_CENTER_PULSE = Math.round((this.TILT_MAX_DOWN_PULSE + this.TILT_MAX_UP_PULSE) / 2);
    this.PAN_CENTER_PULSE = Math.round((this.PAN_MAX_RIGHT_PULSE + this.PAN_MAX_LEFT_PULSE) / 2);

    this.panChannel = 0;
    this.tiltChannel = 1;

    // Current position tracking
    this.currentPosition = {
      pan: this.PAN_CENTER_PULSE,
      tilt: this.TILT_CENTER_PULSE,
    };

    // Processing state
    this.isProcessing = {
      pan: false,
      tilt: false,
    };

    // Request management
    this.queueTimestamp = {
      pan: Date.now(),
      tilt: Date.now(),
    };
    
    this.lastProcessedRequest = {
      pan: null,
      tilt: null,
    };

    // Pending relative movements
    this.pendingRelativeMovements = {
      pan: 0,
      tilt: 0,
    };

    // Movement thresholds
    this.MIN_MOVEMENT_THRESHOLD = 20;
    this.DEBOUNCE_TIME = 100;
    this.MAX_QUEUE_SIZE = 10;
    
    // Queues for pan and tilt movements
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

  async moveToPositionAndWait(panPulse, tiltPulse) {
    return new Promise((resolve, reject) => {
      this.moveToPosition(panPulse, tiltPulse);
  
      const checkInterval = setInterval(() => {
        const panDifference = Math.abs(this.currentPosition.pan - panPulse);
        const tiltDifference = Math.abs(this.currentPosition.tilt - tiltPulse);
  
        if (panDifference < this.MIN_MOVEMENT_THRESHOLD && tiltDifference < this.MIN_MOVEMENT_THRESHOLD) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
  
      setTimeout(() => {
        clearInterval(checkInterval);
        reject(new Error('Movement timed out'));
      }, 5000); // Timeout after 5 seconds
    });
  }
  

  startQueueProcessing() {
    this.processQueue('pan');
    this.processQueue('tilt');
  }

  validatePulseRange(type, pulse) {
    if (type === 'pan') {
      return pulse >= this.PAN_MAX_RIGHT_PULSE && pulse <= this.PAN_MAX_LEFT_PULSE;
    } else if (type === 'tilt') {
      return pulse >= this.TILT_MAX_DOWN_PULSE && pulse <= this.TILT_MAX_UP_PULSE;
    }
    return false;
  }

  shouldProcessMovement(type, newPulse) {
    const currentPulse = this.currentPosition[type];
    const pulseDifference = Math.abs(currentPulse - newPulse);
    
    if (pulseDifference < this.MIN_MOVEMENT_THRESHOLD) {
      return false;
    }

    const timeSinceLastMove = Date.now() - this.queueTimestamp[type];
    if (timeSinceLastMove < this.DEBOUNCE_TIME) {
      return false;
    }

    return true;
  }

  optimizeQueue(type) {
    if (this.queues[type].length <= 1) return;

    // If there are relative movements pending, combine them
    const relativeMovements = this.queues[type]
      .filter(task => task.isRelative)
      .reduce((sum, task) => sum + task.relativeDelta, 0);

    if (relativeMovements !== 0) {
      // Calculate the final position after all relative movements
      const finalPosition = this.currentPosition[type] + relativeMovements;
      
      // Validate the final position
      if (this.validatePulseRange(type, finalPosition)) {
        // Replace all queued movements with a single absolute movement
        this.queues[type] = [{
          pulse: finalPosition,
          isRelative: false,
          originalRelative: relativeMovements
        }];
      } else {
        console.log(`${type.toUpperCase()}: Combined movement out of range`);
        this.queues[type] = []; // Clear invalid movements
      }
    } else {
      // If no relative movements, just keep the latest absolute movement
      const latestRequest = this.queues[type].pop();
      this.queues[type] = [latestRequest];
    }
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
          this.optimizeQueue(type);
          
          if (this.queues[type].length > 0) {
            const task = this.queues[type].shift();
            const currentTime = Date.now();

            if (this.shouldProcessMovement(type, task.pulse)) {
              await this.executeServoCommand(
                type === 'pan' ? this.panChannel : this.tiltChannel,
                task.pulse
              );
              this.queueTimestamp[type] = currentTime;
              this.lastProcessedRequest[type] = {
                ...task,
                timestamp: currentTime
              };
            }
          }
        }
        
        await this.delay(50);
      } catch (error) {
        console.error(`Error processing ${type} queue:`, error);
        await this.delay(1000);
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
      if (!this.validatePulseRange('tilt', pulse)) {
        console.log('TILT: Pulse out of range');
        return;
      }
      this.currentPosition.tilt = pulse;
    } else if (channel === this.panChannel) {
      if (!this.validatePulseRange('pan', pulse)) {
        console.log('PAN: Pulse out of range');
        return;
      }
      this.currentPosition.pan = pulse;
    }

    try {
      await this.pwm.setPulseLength(channel, pulse);
      await this.delay(100);
    } catch (error) {
      console.error('Error setting pulse length:', error);
      throw error;
    }
  }

  moveToPosition(panPulse, tiltPulse) {
    if (panPulse !== null && panPulse !== undefined) {
      if (this.queues.pan.length < this.MAX_QUEUE_SIZE) {
        this.queues.pan.push({ 
          pulse: panPulse,
          isRelative: false
        });
      } else {
        console.log('Pan queue full, dropping request');
      }
    }

    if (tiltPulse !== null && tiltPulse !== undefined) {
      if (this.queues.tilt.length < this.MAX_QUEUE_SIZE) {
        this.queues.tilt.push({ 
          pulse: tiltPulse,
          isRelative: false
        });
      } else {
        console.log('Tilt queue full, dropping request');
      }
    }
  }

  moveToPositionRelative(panPulseRel, tiltPulseRel) {
    console.log('panPulseRel, tiltPulseRel',panPulseRel, tiltPulseRel )
    if (panPulseRel) {
      if (this.queues.pan.length < this.MAX_QUEUE_SIZE) {
        console.log('panPulseRelpanPulseRel', panPulseRel, 'this.currentPosition.pan', this.currentPosition.pan)
        // Add relative movement to queue
        this.queues.pan.push({ 
          pulse: this.currentPosition.pan + 10,
          isRelative: true,
          relativeDelta: 10
        });
      }
    }

    if (tiltPulseRel) {
      if (this.queues.tilt.length < this.MAX_QUEUE_SIZE) {
        console.log('tiltPulseReltiltPulseRel', tiltPulseRel, 'this.currentPosition.pan', this.currentPosition.pan)
        // Add relative movement to queue
        this.queues.tilt.push({ 
          pulse: this.currentPosition.tilt + 10,
          isRelative: true,
          relativeDelta: 10
        });
      }
    }
  }

  centerServos() {
    this.moveToPosition(this.PAN_CENTER_PULSE, this.TILT_CENTER_PULSE);
  }

  getQueueStatus() {
    // Calculate total pending relative movements
    const pendingRelativeMovements = {
      pan: this.queues.pan
        .filter(task => task.isRelative)
        .reduce((sum, task) => sum + task.relativeDelta, 0),
      tilt: this.queues.tilt
        .filter(task => task.isRelative)
        .reduce((sum, task) => sum + task.relativeDelta, 0)
    };

    return {
      panQueueLength: this.queues.pan.length,
      tiltQueueLength: this.queues.tilt.length,
      lastPanMove: this.lastProcessedRequest.pan,
      lastTiltMove: this.lastProcessedRequest.tilt,
      currentPosition: this.currentPosition,
      pendingRelativeMovements
    };
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
}

// Export a singleton instance
const servoSystem = new ServoSystem();
module.exports = servoSystem;
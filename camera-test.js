const NodeWebcam = require('node-webcam');

const opts = {
  width: 4608,
  height: 2592,
  delay: 0,
  saveShots: true,
  output: 'jpeg',
  device: '/dev/video0',
  callbackReturn: 'location',
  verbose: false
};

const Webcam = NodeWebcam.create(opts);

Webcam.capture('test_picture', (err, data) => {
  if (err) {
    console.log(err);
  } else {
    console.log('Image captured:', data);
  }
});

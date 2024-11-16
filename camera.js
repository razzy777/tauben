// camera.js
const { spawn } = require('child_process');

let videoProcess = null;

function startVideoStream(frontendNamespace, aiNamespace) {
  if (videoProcess) {
    console.log('Video stream already running');
    return;
  }

  console.log('Starting video stream...');
  console.log('Frontend namespace clients:', frontendNamespace.sockets.size);
  console.log('AI namespace clients:', aiNamespace.sockets.size);

  const command = 'ffmpeg';
  const args = [
	'-f', 'v4l2',        // Capture from video4linux2
	'-input_format', 'mjpeg', // Input format
	'-video_size', '640x480',
	'-i', '/dev/video0', // Adjust if your camera is on a different device
	'-c:v', 'copy',      // Copy the video codec
	'-f', 'image2pipe',  // Output format
	'-'
  ];
  

  console.log('Executing command:', command, args.join(' '));

  try {
    // Check if libcamera-vid exists
    const { execSync } = require('child_process');
    try {
      execSync('which libcamera-vid');
      console.log('libcamera-vid found');
    } catch (error) {
      console.error('libcamera-vid not found in PATH');
      return;
    }

    // Start video process
    videoProcess = spawn(command, args);
    console.log('Video process started with PID:', videoProcess.pid);

    let buffer = Buffer.from([]);
    let frameCount = 0;

    // Handle stdout data
	let frameBuffer = Buffer.from([]);

	videoProcess.stdout.on('data', (data) => {
	  frameBuffer = Buffer.concat([frameBuffer, data]);
	
	  try {
		while (frameBuffer.length > 0) {
		  // Assume frames are separated by EOF (this may vary depending on the ffmpeg output)
		  const eofIndex = frameBuffer.indexOf(Buffer.from([0xFF, 0xD9]));
		  if (eofIndex === -1) break;
	
		  const frame = frameBuffer.slice(0, eofIndex + 2); // Include EOF
		  frameBuffer = frameBuffer.slice(eofIndex + 2);
	
		  // Emit the frame to frontend and AI namespaces
		  const base64Frame = frame.toString('base64');
		  frontendNamespace.emit('videoFrame', base64Frame);
		  if (frameCount % 15 === 0) {
			aiNamespace.emit('videoFrame', base64Frame);
		  }
	
		  frameCount++;
		}
	  } catch (error) {
		console.error('Error processing video frame:', error);
	  }
	});
	
    // Handle stderr (camera info and errors)
    videoProcess.stderr.on('data', (data) => {
      console.log('Camera stderr:', data.toString());
    });

    // Handle process errors
    videoProcess.on('error', (error) => {
      console.error('Camera process error:', error);
      videoProcess = null;
    });

    // Handle process exit
    videoProcess.on('exit', (code, signal) => {
      console.log('Camera process exited with code:', code, 'signal:', signal);
      videoProcess = null;
    });

  } catch (error) {
    console.error('Failed to start video stream:', error);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
    videoProcess = null;
  }
}

function stopVideoStream() {
  if (videoProcess) {
    console.log('Stopping video stream...');
    try {
      videoProcess.kill('SIGTERM');
      console.log('Video process terminated');
    } catch (error) {
      console.error('Error stopping video stream:', error);
    }
    videoProcess = null;
  } else {
    console.log('No video stream to stop');
  }
}

module.exports = {
  startVideoStream,
  stopVideoStream
};
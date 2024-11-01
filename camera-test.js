const NodeWebcam = require('node-webcam')
const path = require('path') // Import the path module

const opts = {
	width: 4608,
	height: 2592,
	delay: 0,
	saveShots: true,
	output: 'jpeg',
	device: '/dev/video0',
	callbackReturn: 'location',
	verbose: false,
}

const Webcam = NodeWebcam.create(opts)

// Define the folder path
const folderPath = './images' // Replace with your desired folder path

// Ensure the directory exists
const fs = require('fs')
if (!fs.existsSync(folderPath)) {
	fs.mkdirSync(folderPath, { recursive: true })
}

// Capture and save the image in the folder
Webcam.capture(path.join(folderPath, 'test_picture'), (err, data) => {
	if (err) {
		console.log(err)
	} else {
		console.log('Image captured:', data)
	}
})

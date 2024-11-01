const NodeWebcam = require('node-webcam')
const path = require('path')
const fs = require('fs')

const opts = {
	width: 1920, // or 1280
	height: 1080, // or 720
	delay: 0,
	saveShots: true,
	output: 'jpeg',
	device: '/dev/video0',
	callbackReturn: 'location',
	verbose: true, // Enable verbose to get more information
}

const Webcam = NodeWebcam.create(opts)

// Define the folder path
const folderPath = './images'

// Ensure the directory exists
if (!fs.existsSync(folderPath)) {
	console.log('Directory does not exist, creating:', folderPath)
	fs.mkdirSync(folderPath, { recursive: true })
} else {
	console.log('Directory exists:', folderPath)
}

// Full path to save the image
const imagePath = path.join(folderPath, 'test_picture.jpg')

// Capture and save the image in the folder
Webcam.capture(imagePath, (err, data) => {
	if (err) {
		console.error('Error capturing image:', err)
	} else {
		console.log('Image captured and saved at:', data)
	}
})

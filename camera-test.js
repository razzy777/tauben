const { exec } = require('child_process')
const path = require('path')
const fs = require('fs')

// Define the folder path and image file path
const folderPath = '/home/johannes/tauben/images'
const imagePath = path.join(folderPath, 'test_picture.jpg')

// Ensure the directory has correct permissions
if (!fs.existsSync(folderPath)) {
	console.log(`Creating directory at: ${folderPath}`)
	fs.mkdirSync(folderPath, { recursive: true })
	// Set full permissions to avoid permission issues
	fs.chmodSync(folderPath, 0o777)
} else {
	console.log(`Directory exists at: ${folderPath}`)
	fs.chmodSync(folderPath, 0o777) // Ensure permissions are set each time
}

// Capture image using `libcamera-still`
function captureImage() {
	// Command to capture image
	const captureCommand = `libcamera-still -o ${imagePath} -t 1000 --width 1280 --height 720`

	// Execute the capture command
	exec(captureCommand, (err, stdout, stderr) => {
		if (err) {
			console.error('Error capturing image:', err)
			return
		}
		if (stderr) {
			console.error('libcamera-still error:', stderr)
		}
		console.log('Image successfully captured and saved at:', imagePath)
	})
}

// Capture an image
captureImage()

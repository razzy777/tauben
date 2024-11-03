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

// Function to capture image
async function captureImage() {
	return new Promise((resolve, reject) => {
		// Command to capture image
		const captureCommand = `libcamera-still -o ${imagePath} -t 1000 --width 1280 --height 720`

		// Execute the capture command
		exec(captureCommand, (err, stdout, stderr) => {
			if (err) {
				reject(new Error(`Error capturing image: ${err.message}`))
				return
			}
			if (stderr) {
				console.error('libcamera-still error:', stderr)
			}
			console.log('Image successfully captured and saved at:', imagePath)
			resolve(imagePath)
		})
	})
}

// Function to remove the captured image
async function removeImage() {
	return new Promise((resolve, reject) => {
		if (fs.existsSync(imagePath)) {
			fs.unlink(imagePath, (err) => {
				if (err) {
					reject(new Error(`Error removing image: ${err.message}`))
					return
				}
				console.log('Image successfully removed:', imagePath)
				resolve()
			})
		} else {
			reject(new Error('Image file does not exist.'))
		}
	})
}

// Export the functions
module.exports = {
	captureImage,
	removeImage
}

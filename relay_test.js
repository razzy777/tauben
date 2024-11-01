const Gpio = require('onoff').Gpio

try {
	const relay = new Gpio(17, 'out')
	relay.writeSync(1) // Activate relay
	setTimeout(() => {
		relay.writeSync(0) // Deactivate relay after 3 seconds
		relay.unexport()
		console.log('Relay test completed')
	}, 3000)
} catch (error) {
	console.error('Error with GPIO:', error)
}

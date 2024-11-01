const gpiod = require('node-gpiod');

async function main() {
    try {
        console.log('Initializing GPIO...');
        // Open the chip
        const chip = new gpiod.Chip('gpiochip0');
        
        // Get line 17 and set it as output
        const line = chip.getLine(17);
        await line.requestOutput({ consumer: 'relay_test' });
        
        console.log('Activating relay...');
        // Set high
        await line.setValue(1);
        
        // Wait 3 seconds
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        console.log('Deactivating relay...');
        // Set low
        await line.setValue(0);
        
        // Release the line
        await line.release();
        chip.close();
        
        console.log('Relay test completed');
    } catch (error) {
        console.error('Error:', error);
    }
}

// Handle Ctrl+C
process.on('SIGINT', () => {
    try {
        const chip = new gpiod.Chip('gpiochip0');
        const line = chip.getLine(17);
        line.setValue(0);
        line.release();
        chip.close();
    } catch (error) {
        // Ignore cleanup errors
    }
    process.exit();
});

main();
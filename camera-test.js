const PiCamera = require('pi-camera');
const path = require('path');
const fs = require('fs');

class CameraController {
   constructor(options = {}) {
       // Default options
       this.options = {
           width: options.width || 1920,
           height: options.height || 1080,
           outputDir: options.outputDir || './images',
           nopreview: true,
           rotation: options.rotation || 0,
           quality: options.quality || 100,
       };

       // Ensure output directory exists
       if (!fs.existsSync(this.options.outputDir)) {
           console.log('Creating output directory:', this.options.outputDir);
           fs.mkdirSync(this.options.outputDir, { recursive: true });
       }
   }

   async takePhoto(filename) {
       try {
           const outputPath = path.join(this.options.outputDir, filename);
           
           const camera = new PiCamera({
               mode: 'photo',
               output: outputPath,
               width: this.options.width,
               height: this.options.height,
               nopreview: this.options.nopreview,
               rotation: this.options.rotation,
               quality: this.options.quality
           });

           console.log('Taking photo...');
           await camera.snap();
           console.log('Photo saved to:', outputPath);
           return outputPath;
       } catch (error) {
           console.error('Error taking photo:', error);
           throw error;
       }
   }
}

// Example usage
async function testCamera() {
   const camera = new CameraController({
       width: 1920,
       height: 1080,
       outputDir: './captures'
   });

   try {
       const photoPath = await camera.takePhoto(`photo_${Date.now()}.jpg`);
       console.log('Photo captured:', photoPath);
   } catch (error) {
       console.error('Camera test failed:', error);
   }
}

// Run the test
testCamera().catch(console.error);
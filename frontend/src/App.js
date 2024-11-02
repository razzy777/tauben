import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function App() {
  const [detection, setDetection] = useState(null);
  const socket = io('http://192.168.68.56:3000');

  useEffect(() => {
    socket.on('detection', (data) => {
      setDetection(data);
    });

    return () => {
      socket.off('detection');
    };
  }, []);

  const takePhoto = () => {
    socket.emit('takePhoto');
  };

  const moveServo = (pan, tilt) => {
    socket.emit('moveServo', { pan, tilt });
  };

  const centerServo = () => {
    socket.emit('centerServo');
  };

  const activateWater = (duration) => {
    socket.emit('activateWater', duration);
  };

  return (
    <div>
      <h1>System Control Panel</h1>
      <div>
        <button onClick={takePhoto}>Take Photo</button>
        <button onClick={centerServo}>Center Servos</button>
        <button onClick={() => moveServo(1200, 1350)}>Move Servo: Right & Down</button>
        <button onClick={() => moveServo(2300, 2500)}>Move Servo: Left & Up</button>
        <button onClick={() => activateWater(500)}>Activate Water (500ms)</button>
      </div>
      {detection ? (
        <div>
          <p>{detection.detected ? 'Object detected!' : 'No objects detected.'}</p>
          <p>Timestamp: {new Date(detection.timestamp).toLocaleTimeString()}</p>
          {detection.image && (
            <img src={`data:image/jpeg;base64,${detection.image}`} alt="Capture" />
          )}
        </div>
      ) : (
        <p>Waiting for data...</p>
      )}
    </div>
  );
}

export default App;

import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Initialize the socket connection
    const newSocket = io('http://localhost:3000'); // Update with the correct backend address
    
    // Set the socket in the state
    setSocket(newSocket);

    // Attach listener to socket once it is available
    newSocket.on('detection', (data) => {
      setDetection(data);
    });

    // Clean up the socket connection
    return () => {
      newSocket.off('detection');
      newSocket.close();
    };
  }, []); // Empty dependency array so it only runs once

  // Only use socket when it is successfully connected
  const takePhoto = () => {
    if (socket) {
      socket.emit('takePhoto');
    } else {
      console.error('Socket is not available');
    }
  };

  const moveServo = (pan, tilt) => {
    if (socket) {
      socket.emit('moveServo', { pan, tilt });
    } else {
      console.error('Socket is not available');
    }
  };

  const centerServo = () => {
    if (socket) {
      socket.emit('centerServo');
    } else {
      console.error('Socket is not available');
    }
  };

  const activateWater = (duration) => {
    if (socket) {
      socket.emit('activateWater', duration);
    } else {
      console.error('Socket is not available');
    }
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

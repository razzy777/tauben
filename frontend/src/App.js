import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = io('http://192.168.68.56:3000');
    setSocket(newSocket);

    newSocket.on('detection', (data) => {
      setDetection(data);
    });

    return () => {
      newSocket.off('detection');
      newSocket.close();
    };
  }, []);

  socket.on('connect', () => {
    console.log('Successfully connected to the backend');
  });
  
  socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
  });
  
  socket.on('disconnect', () => {
    console.log('Disconnected from server');
  });
  

  const takePhoto = () => {
    if (socket) socket.emit('takePhoto');
  };

  const moveServo = (pan, tilt) => {
    if (socket) socket.emit('moveServo', { pan, tilt });
  };

  const centerServo = () => {
    if (socket) socket.emit('centerServo');
  };

  const activateWater = (duration) => {
    if (socket) socket.emit('activateWater', duration);
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

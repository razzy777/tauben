import React, { useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';
import { Camera, RotateCw, Droplets } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);
  const [keyStates, setKeyStates] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);

  // Constants for movement
  const MOVEMENT_AMOUNT = 10;
  const KEY_REPEAT_DELAY = 100; // ms between repeated movements when key is held

  useEffect(() => {
    const newSocket = io('http://192.168.68.58:3000');
    setSocket(newSocket);

    newSocket.on('detection', (data) => {
      setDetection(data);
    });

    newSocket.on('servoStatus', (status) => {
      setSystemStatus(status);
    });

    return () => {
      newSocket.off('detection');
      newSocket.off('servoStatus');
      newSocket.close();
    };
  }, []);

  const moveServoRelative = useCallback((pan, tilt) => {
    if (socket) {
      socket.emit('moveServoRelative', { pan, tilt });
    }
  }, [socket]);

  // Keyboard control setup with key holding support
  useEffect(() => {
    const keyIntervals = {};
    
    const handleKeyDown = (e) => {
      if (keyStates[e.key]) return; // Already pressed
      
      setKeyStates(prev => ({ ...prev, [e.key]: true }));
      
      const moveBasedOnKey = () => {
        switch (e.key) {
          case 'ArrowUp':
            moveServoRelative(0, MOVEMENT_AMOUNT);
            break;
          case 'ArrowDown':
            moveServoRelative(0, -MOVEMENT_AMOUNT);
            break;
          case 'ArrowLeft':
            moveServoRelative(MOVEMENT_AMOUNT, 0);
            break;
          case 'ArrowRight':
            moveServoRelative(-MOVEMENT_AMOUNT, 0);
            break;
        }
      };

      // Initial movement
      moveBasedOnKey();
      
      // Setup interval for repeated movement
      keyIntervals[e.key] = setInterval(moveBasedOnKey, KEY_REPEAT_DELAY);
    };

    const handleKeyUp = (e) => {
      setKeyStates(prev => ({ ...prev, [e.key]: false }));
      if (keyIntervals[e.key]) {
        clearInterval(keyIntervals[e.key]);
        delete keyIntervals[e.key];
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      Object.values(keyIntervals).forEach(interval => clearInterval(interval));
    };
  }, [moveServoRelative, keyStates]);

  const takePhoto = () => {
    if (socket) socket.emit('takePhoto');
  };

  const centerServo = () => {
    if (socket) socket.emit('centerServo');
  };

  const activateWater = () => {
    if (socket) socket.emit('activateWater', 500);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center">System Control Panel</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Control Grid */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              {/* Top row */}
              <div></div>
              <button
                onClick={() => moveServoRelative(0, MOVEMENT_AMOUNT)}
                className={`p-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg flex items-center justify-center ${
                  keyStates['ArrowUp'] ? 'bg-blue-700' : ''
                }`}
              >
                ▲
              </button>
              <div></div>

              {/* Middle row */}
              <button
                onClick={() => moveServoRelative(MOVEMENT_AMOUNT, 0)}
                className={`p-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg flex items-center justify-center ${
                  keyStates['ArrowLeft'] ? 'bg-blue-700' : ''
                }`}
              >
                ◄
              </button>
              <button
                onClick={centerServo}
                className="p-4 bg-gray-500 hover:bg-gray-600 text-white rounded-lg flex items-center justify-center"
              >
                <RotateCw className="w-6 h-6" />
              </button>
              <button
                onClick={() => moveServoRelative(-MOVEMENT_AMOUNT, 0)}
                className={`p-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg flex items-center justify-center ${
                  keyStates['ArrowRight'] ? 'bg-blue-700' : ''
                }`}
              >
                ►
              </button>

              {/* Bottom row */}
              <div></div>
              <button
                onClick={() => moveServoRelative(0, -MOVEMENT_AMOUNT)}
                className={`p-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg flex items-center justify-center ${
                  keyStates['ArrowDown'] ? 'bg-blue-700' : ''
                }`}
              >
                ▼
              </button>
              <div></div>
            </div>

            {/* Action buttons */}
            <div className="flex justify-center gap-4">
              <button
                onClick={takePhoto}
                className="p-4 bg-green-500 hover:bg-green-600 text-white rounded-lg flex items-center justify-center gap-2"
              >
                <Camera className="w-6 h-6" />
                Take Photo
              </button>
              <button
                onClick={activateWater}
                className="p-4 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg flex items-center justify-center gap-2"
              >
                <Droplets className="w-6 h-6" />
                Activate Water
              </button>
            </div>

            {/* Status Display */}
            {systemStatus && (
              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-semibold mb-2">System Status</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>Pan Queue: {systemStatus.panQueueLength}</div>
                  <div>Tilt Queue: {systemStatus.tiltQueueLength}</div>
                  <div>Pan Position: {systemStatus.currentPosition?.pan}</div>
                  <div>Tilt Position: {systemStatus.currentPosition?.tilt}</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Photo Display */}
        {detection && (
          <Card>
            <CardHeader>
              <CardTitle>Latest Capture</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-gray-600">
                  Timestamp: {new Date(detection.timestamp).toLocaleTimeString()}
                </p>
                {detection.image && (
                  <img
                    src={`data:image/jpeg;base64,${detection.image}`}
                    alt="Capture"
                    className="w-full rounded-lg shadow-lg"
                  />
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;
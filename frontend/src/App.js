import React, { useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);
  const [keyStates, setKeyStates] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);

  const MOVEMENT_AMOUNT = 10;
  const KEY_REPEAT_DELAY = 100;

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

  useEffect(() => {
    const keyIntervals = {};
    
    const handleKeyDown = (e) => {
      if (keyStates[e.key]) return;
      
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

      moveBasedOnKey();
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

  // Custom button component for better consistency
  const ControlButton = ({ onClick, children, active, color = "blue" }) => (
    <button
      onClick={onClick}
      className={`
        w-16 h-16 rounded-xl font-bold text-2xl
        flex items-center justify-center
        shadow-lg active:shadow-md
        transform active:scale-95 transition-all duration-150
        ${active ? `bg-${color}-600 hover:bg-${color}-700` : `bg-${color}-500 hover:bg-${color}-600`}
        text-white
        focus:outline-none focus:ring-2 focus:ring-${color}-400 focus:ring-opacity-50
      `}
    >
      {children}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-12 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
          System Control Panel
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column - Controls */}
          <div className="bg-gray-800 rounded-2xl p-8 shadow-xl backdrop-blur-sm bg-opacity-50">
            {/* Direction Controls */}
            <div className="grid grid-cols-3 gap-4 mb-8 max-w-[280px] mx-auto">
              <div></div>
              <ControlButton 
                onClick={() => moveServoRelative(0, MOVEMENT_AMOUNT)}
                active={keyStates['ArrowUp']}
              >
                ▲
              </ControlButton>
              <div></div>

              <ControlButton
                onClick={() => moveServoRelative(MOVEMENT_AMOUNT, 0)}
                active={keyStates['ArrowLeft']}
              >
                ◄
              </ControlButton>
              <ControlButton
                onClick={centerServo}
                color="gray"
              >
                ⟲
              </ControlButton>
              <ControlButton
                onClick={() => moveServoRelative(-MOVEMENT_AMOUNT, 0)}
                active={keyStates['ArrowRight']}
              >
                ►
              </ControlButton>

              <div></div>
              <ControlButton
                onClick={() => moveServoRelative(0, -MOVEMENT_AMOUNT)}
                active={keyStates['ArrowDown']}
              >
                ▼
              </ControlButton>
              <div></div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-center gap-4 mb-8">
              <button
                onClick={takePhoto}
                className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 rounded-xl
                          text-white font-semibold shadow-lg active:shadow-md
                          transform active:scale-95 transition-all duration-150
                          focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-opacity-50"
              >
                📸 Take Photo
              </button>
              <button
                onClick={activateWater}
                className="px-6 py-3 bg-cyan-500 hover:bg-cyan-600 rounded-xl
                          text-white font-semibold shadow-lg active:shadow-md
                          transform active:scale-95 transition-all duration-150
                          focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-50"
              >
                💧 Water
              </button>
            </div>

            {/* Status Display */}
            {systemStatus && (
              <div className="bg-gray-700 rounded-xl p-4">
                <h3 className="text-lg font-semibold mb-3 text-blue-300">System Status</h3>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                  <div className="bg-gray-800 p-2 rounded">Pan Queue: {systemStatus.panQueueLength}</div>
                  <div className="bg-gray-800 p-2 rounded">Tilt Queue: {systemStatus.tiltQueueLength}</div>
                  <div className="bg-gray-800 p-2 rounded">Pan: {systemStatus.currentPosition?.pan}</div>
                  <div className="bg-gray-800 p-2 rounded">Tilt: {systemStatus.currentPosition?.tilt}</div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Photo Display */}
          <div className="bg-gray-800 rounded-2xl p-8 shadow-xl backdrop-blur-sm bg-opacity-50">
            <h2 className="text-2xl font-bold mb-4 text-blue-300">Camera Feed</h2>
            {detection ? (
              <div className="space-y-4">
                <p className="text-gray-400">
                  Captured at: {new Date(detection.timestamp).toLocaleTimeString()}
                </p>
                {detection.image && (
                  <div className="relative group">
                    <img
                      src={`data:image/jpeg;base64,${detection.image}`}
                      alt="Capture"
                      className="w-full h-auto rounded-xl shadow-lg"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-500">
                No image captured yet
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
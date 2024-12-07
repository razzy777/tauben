import React, { useEffect, useState, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import styled from 'styled-components';
import debounce from 'lodash/debounce';

// Styled Components
const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a1a 0%, #0a192f 100%);
  padding: 2rem;
  color: white;
`;

const Panel = styled.div`
  max-width: 1600px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const LiveFeedContainer = styled.div`
  position: relative;
  width: 100%;
  background: #000;
  margin-bottom: 1rem;
  overflow: hidden;

  &::before {
    content: "";
    display: block;
    padding-top: 100% /*56.25%; 16:9 Aspect Ratio */
  }
  max-width: 640px;
  margin-left: auto;
  margin-right: auto;

`;

const VideoOverlayContainer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const Video = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const Title = styled.h1`
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 2rem;
  color: #64ffda;
  text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
`;

const ControlsWrapper = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ControlCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const ControlGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  max-width: 300px;
  margin: 0 auto 2rem auto;
`;

const DirectionButton = styled.button`
  background: ${props => props.active ? '#3b82f6' : '#2563eb'};
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 1rem;
  font-size: 1.25rem;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

  &:hover {
    background: #1d4ed8;
    transform: translateY(-2px);
  }

  &:active {
    transform: translateY(0);
  }

  ${props => props.center && `
    background: #4b5563;
    &:hover {
      background: #374151;
    }
  `}
`;

const ActionButton = styled.button`
  background: ${props => props.water ? '#06b6d4' : '#10b981'};
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

  &:hover {
    background: ${props => props.water ? '#0891b2' : '#059669'};
    transform: translateY(-2px);
  }

  &:active {
    transform: translateY(0);
  }
`;

const ActionButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const StatusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  background: rgba(0, 0, 0, 0.2);
  padding: 1rem;
  border-radius: 0.5rem;
`;

const StatusItem = styled.div`
  background: rgba(0, 0, 0, 0.3);
  padding: 0.75rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #94a3b8;
`;

const NoImage = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 0.5rem;
  color: #94a3b8;
`;

const BoundingBox = styled.div`
  position: absolute;
  border: 2px solid red;
  background-color: rgba(255, 0, 0, 0.3);
`;

const Label = styled.div`
  position: absolute;
  top: -25px;
  left: -2px; // Align with border
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 3px 8px;
  font-size: 12px;
  white-space: nowrap;
  border-radius: 3px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 6px;

  &::after {
    content: "${props => props.confidence}%";
    background-color: rgba(255, 255, 255, 0.2);
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 11px;
  }
`;

const ConnectionStatus = styled.div`
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  background: ${props => props.connected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'};
  color: ${props => props.connected ? '#10b981' : '#ef4444'};
  backdrop-filter: blur(10px);
  z-index: 1000;
`;

function App() {
  const [videoFrame, setVideoFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [buttonStates, setButtonStates] = useState({});
  const [keysPressed, setKeysPressed] = useState({});
  const socketRef = useRef(null);
  
  const MOVEMENT_AMOUNT = 5;

  const moveServoRelative = useCallback((pan, tilt) => {
    if (socketRef.current) {
      socketRef.current.emit('moveServoRelative', { pan, tilt });
    }
  }, []);

  // Socket connection setup
  useEffect(() => {
    const cleanup = () => {
      if (socketRef.current) {
        console.log('Cleaning up socket connection');
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };

    if (!socketRef.current) {
      console.log('Initializing socket connection');
      
      const socket = io('http://192.168.68.68:3000/frontend', {
        transports: ['websocket'],
        upgrade: false,
        forceNew: true,
        reconnection: false,
        timeout: 5000,
        autoConnect: false,
        maxPayload: 64000
      });

      socket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        setIsConnected(false);
        cleanup();
      });

      socket.on('connect', () => {
        console.log('Socket connected');
        setIsConnected(true);
      });

      socket.on('disconnect', () => {
        console.log('Socket disconnected');
        setIsConnected(false);
        cleanup();
      });

      let lastFrameTime = Date.now();
      const FRAME_THROTTLE = 100;

      socket.on('videoFrame', (frameData) => {
        const now = Date.now();
        if (now - lastFrameTime >= FRAME_THROTTLE) {
          setVideoFrame(frameData);
          lastFrameTime = now;
        }
      });

      socket.on('detections', (data) => {
        setDetections(data);
      });

      socketRef.current = socket;

      try {
        socket.connect();
      } catch (error) {
        console.error('Failed to connect socket:', error);
        cleanup();
      }
    }

    return cleanup;
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        setKeysPressed(prev => ({ ...prev, [e.key]: true }));
      }
    };

    const handleKeyUp = (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        setKeysPressed(prev => ({ ...prev, [e.key]: false }));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Servo movement handling
  useEffect(() => {
    let lastMoveTime = 0;
    const throttleDelay = 100;

    const moveServos = (timestamp) => {
      if (!lastMoveTime || timestamp - lastMoveTime >= throttleDelay) {
        if (keysPressed['ArrowUp']) moveServoRelative(0, MOVEMENT_AMOUNT);
        if (keysPressed['ArrowDown']) moveServoRelative(0, -MOVEMENT_AMOUNT);
        if (keysPressed['ArrowLeft']) moveServoRelative(MOVEMENT_AMOUNT, 0);
        if (keysPressed['ArrowRight']) moveServoRelative(-MOVEMENT_AMOUNT, 0);
        lastMoveTime = timestamp;
      }
      animationFrameId = requestAnimationFrame(moveServos);
    };

    let animationFrameId = requestAnimationFrame(moveServos);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [keysPressed, moveServoRelative]);

  const handleButtonPress = (direction) => {
    setButtonStates(prev => ({ ...prev, [direction]: true }));

    switch (direction) {
      case 'up':
        moveServoRelative(0, MOVEMENT_AMOUNT);
        break;
      case 'down':
        moveServoRelative(0, -MOVEMENT_AMOUNT);
        break;
      case 'left':
        moveServoRelative(MOVEMENT_AMOUNT, 0);
        break;
      case 'right':
        moveServoRelative(-MOVEMENT_AMOUNT, 0);
        break;
      default:
        break;
    }

    setTimeout(() => {
      setButtonStates(prev => ({ ...prev, [direction]: false }));
    }, 100);
  };

  const handleSprayWater = (ms) => {
    if (socketRef.current) {
      socketRef.current.emit('activateWater', ms);
    }
  };

  const handleActivatePump = (ms) => {
    if (socketRef.current) {
      socketRef.current.emit('activatePump', ms);
    }
  };

  return (
    <Container>
      <Title>System Control Panel</Title>
      <Panel>
        <ControlCard>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#64ffda' }}>
            Camera Feed {isConnected ? '(Connected)' : '(Disconnected)'}
          </h2>
          <LiveFeedContainer>
            {isConnected && videoFrame ? (
              <VideoOverlayContainer>
                <Video
                  src={`data:image/jpeg;base64,${videoFrame}`}
                  alt="Live Feed"
                  onError={(e) => console.error('Image failed to load:', e)}
                />
                  {detections.map((detection, index) => {
                    const { box, meta } = detection;
                    const [ymin, xmin, ymax, xmax] = box;
                    const confidence = (meta.confidence * 100).toFixed(1);

                    return (
                      <BoundingBox
                        key={index}
                        style={{
                          top: `${ymin * 100}%`,
                          left: `${xmin * 100}%`,
                          width: `${(xmax - xmin) * 100}%`,
                          height: `${(ymax - ymin) * 100}%`
                        }}
                      >
                        <Label confidence={confidence}>
                          {meta.className}
                        </Label>
                      </BoundingBox>
                    );
                  })}
              </VideoOverlayContainer>
            ) : (
              <NoImage>
                {isConnected ? 'Waiting for video feed...' : 'Disconnected - Trying to reconnect...'}
              </NoImage>
            )}
          </LiveFeedContainer>
        </ControlCard>

        <ControlsWrapper>
          <ControlCard>
            <ControlGrid>
              <div />
              <DirectionButton
                onMouseDown={() => handleButtonPress('up')}
                active={buttonStates['up']}
              >
                ▲
              </DirectionButton>
              <div />

              <DirectionButton
                onMouseDown={() => handleButtonPress('left')}
                active={buttonStates['left']}
              >
                ◄
              </DirectionButton>
              <DirectionButton
                onClick={() => socketRef.current?.emit('centerServo')}
                center
              >
                ⟲
              </DirectionButton>
              <DirectionButton
                onMouseDown={() => handleButtonPress('right')}
                active={buttonStates['right']}
              >
                ►
              </DirectionButton>

              <div />
              <DirectionButton
                onMouseDown={() => handleButtonPress('down')}
                active={buttonStates['down']}
              >
                ▼
              </DirectionButton>
              <div />
            </ControlGrid>

            <ActionButtonContainer>
              <ActionButton onClick={() => socketRef.current?.emit('takePhoto')}>
                📸 Take Photo
              </ActionButton>
              <ActionButton onClick={() => handleSprayWater(10)} water>
                💧 Spray Water (10ms)
              </ActionButton>
              <ActionButton onClick={() => handleSprayWater(50)} water>
                💧 Spray Water (50ms)
              </ActionButton>
              <ActionButton onClick={() => handleActivatePump(1000)} water>
                💧 Pump Water (1 sec)
              </ActionButton>
              <ActionButton onClick={() => handleActivatePump(3000)} water>
              💧 Pump Water (3 sec)
              </ActionButton>
              <ActionButton onClick={() => handleActivatePump(5000)} water>
              💧 Pump Water (5 sec)
              </ActionButton>
              <ActionButton onClick={() => handleActivatePump(10000)} water>
              💧 Pump Water (10 sec)
              </ActionButton>
              <ActionButton onClick={() => socketRef.current?.emit('startScan')}>
                🔍 Scan
              </ActionButton>
            </ActionButtonContainer>
          </ControlCard>
        </ControlsWrapper>
      </Panel>
      <ConnectionStatus connected={isConnected}>
        {isConnected ? 'Connected' : 'Disconnected'}
      </ConnectionStatus>
    </Container>
  );
}

export default App;

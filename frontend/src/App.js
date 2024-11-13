import React, { useEffect, useState, useCallback } from 'react';
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
    padding-top: 56.25%; /* 16:9 Aspect Ratio */
  }
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

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);
  const [keysPressed, setKeysPressed] = useState({});
  const [buttonStates, setButtonStates] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [videoFrame, setVideoFrame] = useState(null);
  const [crosshairPosition, setCrosshairPosition] = useState({ x: 50, y: 50 }); // Start in center
  const [detections, setDetections] = useState([]);

  const MOVEMENT_AMOUNT = 5;

  const moveServoRelative = useCallback((pan, tilt) => {
    if (socket) {
      socket.emit('moveServoRelative', { pan, tilt });
    }
  }, [socket]);

  const moveServoRelativeDebounced = useCallback(
    debounce((pan, tilt) => {
      if (socket) {
        socket.emit('moveServoRelative', { pan, tilt });
      }
    }, 100), // Adjust the delay as needed
    [socket]
  );

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault(); // Prevent default scrolling behavior
        setKeysPressed((prevKeys) => ({
          ...prevKeys,
          [e.key]: true,
        }));
      }
    };

    const handleKeyUp = (e) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        setKeysPressed((prevKeys) => ({
          ...prevKeys,
          [e.key]: false,
        }));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Throttle the servo movement when arrow keys are held down
  useEffect(() => {
    let lastMoveTime = 0;
    const throttleDelay = 100; // milliseconds

    const moveServos = (timestamp) => {
      if (!lastMoveTime || timestamp - lastMoveTime >= throttleDelay) {
        if (keysPressed['ArrowUp']) {
          moveServoRelative(0, MOVEMENT_AMOUNT);
        }
        if (keysPressed['ArrowDown']) {
          moveServoRelative(0, -MOVEMENT_AMOUNT);
        }
        if (keysPressed['ArrowLeft']) {
          moveServoRelative(MOVEMENT_AMOUNT, 0);
        }
        if (keysPressed['ArrowRight']) {
          moveServoRelative(-MOVEMENT_AMOUNT, 0);
        }
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
    // Update the button state to active
    setButtonStates((prev) => ({ ...prev, [direction]: true }));

    // Move the servo
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

    // Reset the button state after a short delay
    setTimeout(() => {
      setButtonStates((prev) => ({ ...prev, [direction]: false }));
    }, 100); // Adjust the delay as needed
  };

  const handleSprayWater = () => {
    if (socket) {
      socket.emit('activateWater', 500); // Adjust duration as needed
    }
  };

  const handleVideoClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    setCrosshairPosition({ x, y });
  };

  const adjustServosToFollow = useCallback((boundingBox) => {
    // boundingBox format: [ymin, xmin, ymax, xmax] normalized between 0 and 1
    const [ymin, xmin, ymax, xmax] = boundingBox;

    // Calculate center of bounding box
    const centerX = (xmin + xmax) / 2;
    const centerY = (ymin + ymax) / 2;

    // Map centerX and centerY to percentages
    const crosshairX = centerX * 100;
    const crosshairY = centerY * 100;

    // Update crosshair position (optional)
    setCrosshairPosition({ x: crosshairX, y: crosshairY });

    // Map crosshair position to servo movement
    const deltaX = crosshairX - 50; // Range -50 to +50
    const deltaY = crosshairY - 50;

    const panMovement = -deltaX * 0.1; // Adjust scaling factor as needed
    const tiltMovement = deltaY * 0.1;

    moveServoRelativeDebounced(panMovement, tiltMovement);
  }, [moveServoRelativeDebounced]);

  // Socket connection and event handlers
  useEffect(() => {
    const newSocket = io('http://192.168.68.68:3000/frontend'); // Adjust the URL if needed
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Socket connected');
    });

    newSocket.on('videoFrame', (frameData) => {
      setVideoFrame(frameData);
    });

    newSocket.on('detections', (data) => {
      console.log('Detections received:', data);
      setDetections(data);

      if (data.length > 0) {
        const personDetection = data.find(d => d.class === 'person');
        if (personDetection) {
          adjustServosToFollow(personDetection.box);
        }
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
    });

    return () => {
      newSocket.close();
    };
  }, [adjustServosToFollow]);

  return (
    <Container>
      <Title>System Control Panel</Title>
      <Panel>
        {/* Main video feed */}
        <ControlCard>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#64ffda' }}>Camera Feed</h2>
          <LiveFeedContainer>
            {videoFrame ? (
              <VideoOverlayContainer onClick={handleVideoClick}>
                <Video
                  src={`data:image/jpeg;base64,${videoFrame}`}
                  alt="Live Feed"
                  onError={(e) => console.error('Image failed to load:', e)}
                />
                {detections.map((detection, index) => {
                  const { box, class: className, score } = detection;
                  const [ymin, xmin, ymax, xmax] = box;

                  return (
                    <BoundingBox
                      key={index}
                      style={{
                        top: `${ymin * 100}%`,
                        left: `${xmin * 100}%`,
                        width: `${(xmax - xmin) * 100}%`,
                        height: `${(ymax - ymin) * 100}%`
                      }}
                    />
                  );
                })}
              </VideoOverlayContainer>
            ) : (
              <NoImage>Waiting for video feed...</NoImage>
            )}
          </LiveFeedContainer>
        </ControlCard>

        {/* Controls section */}
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
                onClick={() => socket?.emit('centerServo')}
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
              <ActionButton onClick={() => socket?.emit('takePhoto')}>
                📸 Take Photo
              </ActionButton>
              <ActionButton
                onClick={handleSprayWater}
                water
              >
                💧 Spray Water
              </ActionButton>
              <ActionButton onClick={() => socket?.emit('startScan')}>
                🔍 Scan
              </ActionButton>
            </ActionButtonContainer>

            {systemStatus && (
              <StatusGrid>
                <StatusItem>Pan Queue: {systemStatus.panQueueLength}</StatusItem>
                <StatusItem>Tilt Queue: {systemStatus.tiltQueueLength}</StatusItem>
                <StatusItem>Pan: {systemStatus.currentPosition?.pan}</StatusItem>
                <StatusItem>Tilt: {systemStatus.currentPosition?.tilt}</StatusItem>
              </StatusGrid>
            )}
          </ControlCard>

          {/* Captures section */}
          <ControlCard>
            <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#64ffda' }}>Captures</h2>
            {detection && detection.image && (
              <div>
                <p style={{ color: '#94a3b8', marginBottom: '1rem' }}>
                  Last capture at: {new Date(detection.timestamp).toLocaleTimeString()}
                </p>
                <img
                  src={`data:image/jpeg;base64,${detection.image}`}
                  alt="Capture"
                  style={{
                    width: '100%',
                    borderRadius: '0.5rem',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                  }}
                />
              </div>
            )}
          </ControlCard>
        </ControlsWrapper>
      </Panel>
    </Container>
  );
}

export default App;

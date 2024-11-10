import React, { useEffect, useState, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import styled from 'styled-components';

// Styled Components
const Container = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a1a 0%, #0a192f 100%);
  padding: 2rem;
  color: white;
`;

const Panel = styled.div`
  max-width: 1600px; // Increased max-width
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

// Adjust LiveFeedContainer to have a 16:9 aspect ratio
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

// Ensure VideoOverlayContainer fills the parent
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


const Crosshair = styled.div`
  position: absolute;
  width: 40px;
  height: 40px;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 10;
  left: ${props => props.x}%;
  top: ${props => props.y}%;

  /* Crosshair styles */
  &::before,
  &::after {
    content: '';
    position: absolute;
    background-color: rgba(255, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.7);
  }

  &::before {
    top: 50%;
    left: 0;
    right: 0;
    height: 2px;
    transform: translateY(-50%);
  }

  &::after {
    left: 50%;
    top: 0;
    bottom: 0;
    width: 2px;
    transform: translateX(-50%);
  }

  .circle-outer {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 40px;
    height: 40px;
    border: 2px solid rgba(255, 0, 0, 0.7);
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }

  .circle-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 10px;
    height: 10px;
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    background-color: rgba(255, 0, 0, 0.3);
  }
`;


const BoundingBox = styled.div`
  position: absolute;
  border: 2px solid red;
  background-color: rgba(255, 0, 0, 0.3);
`;




function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);
  const [keyStates, setKeyStates] = useState({});
  const [keysPressed, setKeysPressed] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [videoFrame, setVideoFrame] = useState(null);
  const [crosshairPosition, setCrosshairPosition] = useState({ x: 50, y: 50 }); // Start in center
  const [detections, setDetections] = useState([]);
  const crosshairStep = 1; // Amount to move crosshair per keypress
  



  const MOVEMENT_AMOUNT = 5;

  const moveCrosshair = useCallback((direction) => {
    setCrosshairPosition(prev => {
        const newPos = { ...prev };
        switch (direction) {
            case 'up':
                newPos.y = Math.max(0, prev.y - crosshairStep);
                break;
            case 'down':
                newPos.y = Math.min(100, prev.y + crosshairStep);
                break;
            case 'left':
                newPos.x = Math.max(0, prev.x - crosshairStep);
                break;
            case 'right':
                newPos.x = Math.min(100, prev.x + crosshairStep);
                break;
            default:
                break;
        }
        return newPos;
    });
  }, [crosshairStep]);

  useEffect(() => {
    const handleKeyDown = (e) => {
        switch(e.key) {
            case 'w':
                moveCrosshair('up');
                break;
            case 's':
                moveCrosshair('down');
                break;
            case 'a':
                moveCrosshair('left');
                break;
            case 'd':
                moveCrosshair('right');
                break;
            default:
                break;
        }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [moveCrosshair]);

  const mapCrosshairPositionToServoPulse = (crosshairX, crosshairY, currentPanPulse, currentTiltPulse) => {
    // Constants
    const PAN_MIN_ANGLE = -90;
    const PAN_MAX_ANGLE = 90;
    const TILT_MIN_ANGLE = -70; // Adjust based on total tilt range
    const TILT_MAX_ANGLE = 70;  // Adjust based on total tilt range
  
    // Servo pulse ranges
    const PAN_MAX_LEFT_PULSE = 2000;
    const PAN_MAX_RIGHT_PULSE = 1200;
    const TILT_MAX_UP_PULSE = 1350;   // Up corresponds to minimum pulse
    const TILT_MAX_DOWN_PULSE = 2400; // Down corresponds to maximum pulse
  
    // Camera field of view
    const CAMERA_FOV_HORIZONTAL = 60; // degrees
    const CAMERA_FOV_VERTICAL = 40;   // degrees
  
    // Calculate angular offsets from the center
    const panAngleOffset = ((crosshairX - 50) / 50) * (CAMERA_FOV_HORIZONTAL / 2);
    const tiltAngleOffset = ((crosshairY - 50) / 50) * (CAMERA_FOV_VERTICAL / 2);
  
    // Desired pan and tilt angles (assuming current angle is 0)
    const desiredPanAngle = panAngleOffset;
    const desiredTiltAngle = -tiltAngleOffset; // Negative because Y-axis might be inverted
  
    // Map desired angles to servo pulses
    const panPulseDesired =
      PAN_MAX_LEFT_PULSE +
      ((desiredPanAngle - PAN_MIN_ANGLE) * (PAN_MAX_RIGHT_PULSE - PAN_MAX_LEFT_PULSE)) /
        (PAN_MAX_ANGLE - PAN_MIN_ANGLE);
  
    const tiltPulseDesired =
      TILT_MAX_UP_PULSE +
      ((desiredTiltAngle - TILT_MIN_ANGLE) * (TILT_MAX_DOWN_PULSE - TILT_MAX_UP_PULSE)) /
        (TILT_MAX_ANGLE - TILT_MIN_ANGLE);
  
    // Compute the difference from current pulses
    let panPulseDelta = panPulseDesired - currentPanPulse;
    let tiltPulseDelta = tiltPulseDesired - currentTiltPulse;
  
    // Limit the maximum step size for smoother movement
    const MAX_PAN_PULSE_STEP = 10; // Adjust this value as needed
    const MAX_TILT_PULSE_STEP = 10; // Adjust this value as needed
  
    // Clamp the deltas to the maximum step size
    panPulseDelta = Math.max(Math.min(panPulseDelta, MAX_PAN_PULSE_STEP), -MAX_PAN_PULSE_STEP);
    tiltPulseDelta = Math.max(Math.min(tiltPulseDelta, MAX_TILT_PULSE_STEP), -MAX_TILT_PULSE_STEP);
    console.log('DDAAA PAN', panPulseDelta)
    console.log('DDAAA TILT', tiltPulseDelta)

    return { panPulseDelta, tiltPulseDelta };
  };
    
  const handleSprayWater = () => {
    const { panPulse, tiltPulse } = mapCrosshairPositionToServoPulse(crosshairPosition.x, crosshairPosition.y);
  
    if (socket) {
      socket.emit('moveToPositionAndSpray', { panPulse, tiltPulse, duration: 500 });
    }
  };

  const adjustServosToFollow = (boundingBox) => {
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
  
    // Map crosshair position to servo pulse
    const { panPulseDelta, tiltPulseDelta } = mapCrosshairPositionToServoPulse(crosshairX, crosshairY);
    console.log('panPulse, ', panPulseDelta, 'TILt' , tiltPulseDelta)


  
    // Move servos to follow the person
    if (socket) {
      socket.emit('moveServoRelative', { pan: panPulseDelta, tilt: tiltPulseDelta });
    }
  };
  



  // Socket connection and event handlers remain the same
  useEffect(() => {
    const newSocket = io('http://192.168.68.68:3000/frontend'); // Connect to /frontend namespace
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
  }, []);
    // Movement handlers remain the same
  const moveServoRelative = useCallback((pan, tilt) => {
    if (socket) {
      socket.emit('moveServoRelative', { pan, tilt });
    }
  }, [socket]);

  const handleVideoClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
  
    setCrosshairPosition({ x, y });
  };
  
  const keyIntervals = useRef({});

  // Keyboard controls remain the same
  useEffect(() => {
    const handleKeyDown = (e) => {
      setKeysPressed((prevKeys) => ({
        ...prevKeys,
        [e.key]: true,
      }));
    };
  
    const handleKeyUp = (e) => {
      setKeysPressed((prevKeys) => ({
        ...prevKeys,
        [e.key]: false,
      }));
    };
  
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
  
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);
  useEffect(() => {
    let animationFrameId;
  
    const moveServos = () => {
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
      animationFrameId = requestAnimationFrame(moveServos);
    };
  
    moveServos();
  
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [keysPressed, moveServoRelative]);
  
    

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
                onClick={() => moveServoRelative(0, MOVEMENT_AMOUNT)}
                active={keyStates['ArrowUp']}
              >
                ▲
              </DirectionButton>
              <div />
              
              <DirectionButton
                onClick={() => moveServoRelative(MOVEMENT_AMOUNT, 0)}
                active={keyStates['ArrowLeft']}
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
                onClick={() => moveServoRelative(-MOVEMENT_AMOUNT, 0)}
                active={keyStates['ArrowRight']}
              >
                ►
              </DirectionButton>
              
              <div />
              <DirectionButton
                onClick={() => moveServoRelative(0, -MOVEMENT_AMOUNT)}
                active={keyStates['ArrowDown']}
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
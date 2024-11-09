import React, { useEffect, useState, useCallback } from 'react';
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

const LiveFeedContainer = styled.div`
    position: relative;
    width: 100%;
    background: #000;
    margin-bottom: 1rem;
    &::before {
        content: "";
        display: block;
        padding-top: 56.25%;
    }
`;

const Video = styled.img`
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
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

const CapturesSection = styled.div`
  margin-top: 2rem;
`;

function App() {
  const [detection, setDetection] = useState(null);
  const [socket, setSocket] = useState(null);
  const [keyStates, setKeyStates] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [videoFrame, setVideoFrame] = useState(null);

  const MOVEMENT_AMOUNT = 10;
  const KEY_REPEAT_DELAY = 100;

  // Socket connection and event handlers remain the same
  useEffect(() => {
    const newSocket = io('http://192.168.68.68:3000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
        console.log('Socket connected');
    });

    newSocket.on('videoFrame', (frameData) => {
        console.log('Received frame, length:', frameData.length);
        setVideoFrame(frameData);
    });

    newSocket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
    });

    return () => {
        console.log('Cleaning up socket connection');
        newSocket.close();
    };
}, []);

  // Movement handlers remain the same
  const moveServoRelative = useCallback((pan, tilt) => {
    if (socket) {
      socket.emit('moveServoRelative', { pan, tilt });
    }
  }, [socket]);

  // Keyboard controls remain the same
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
          default:
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

  return (
    <Container>
      <Title>System Control Panel</Title>
      <Panel>
        {/* Main video feed */}
        <ControlCard>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#64ffda' }}>Camera Feed</h2>
          <LiveFeedContainer>
              {videoFrame ? (
                  <Video
                      src={`data:image/jpeg;base64,${videoFrame}`}
                      alt="Live Feed"
                      onError={(e) => console.error('Video error:', e)}
                      onLoad={() => console.log('Frame loaded successfully')}
                  />
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
                onClick={() => socket?.emit('activateWater', 500)}
                water
              >
                💧 Water
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
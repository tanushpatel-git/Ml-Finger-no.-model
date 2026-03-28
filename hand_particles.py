import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random

class Particle:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = 255
        self.color = (
            random.randint(50, 255),
            random.randint(100, 255),
            random.randint(200, 255)
        )
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        self.vx += dx * 0.02
        self.vy += dy * 0.02
        self.life -= 3

    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / 255
            b, g, r = self.color
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, (int(r * alpha), int(g * alpha), int(b * alpha)), -1)

    def is_alive(self):
        return self.life > 0

def get_number_points(num, width, height):
    points = []
    scale = min(width, height) / 400
    
    if num == 1:
        for y in np.linspace(height * 0.2, height * 0.8, 50):
            points.append((width * 0.4, y))
    elif num == 2:
        for x in np.linspace(width * 0.3, width * 0.7, 30):
            y = height * 0.2 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.3)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 40):
            y = height * 0.5 - ((x - width * 0.3) / (width * 0.4)) * (height * 0.3)
            points.append((x, y))
    elif num == 3:
        for x in np.linspace(width * 0.3, width * 0.7, 25):
            y = height * 0.2 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.15)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 25):
            y = height * 0.35 + ((width * 0.7 - x) / (width * 0.4)) * (height * 0.15)
            points.append((x, y))
        for x in np.linspace(width * 0.3, width * 0.7, 25):
            y = height * 0.5 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.3)
            points.append((x, y))
    elif num == 4:
        for x in np.linspace(width * 0.3, width * 0.7, 20):
            y = height * 0.2 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.1)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 20):
            y = height * 0.3 + ((width * 0.7 - x) / (width * 0.4)) * (height * 0.1)
            points.append((x, y))
        for x in np.linspace(width * 0.3, width * 0.7, 20):
            y = height * 0.4 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.1)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 20):
            y = height * 0.5 + ((width * 0.7 - x) / (width * 0.4)) * (height * 0.3)
            points.append((x, y))
    elif num == 5:
        for x in np.linspace(width * 0.3, width * 0.7, 15):
            y = height * 0.15 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.08)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 15):
            y = height * 0.23 + ((width * 0.7 - x) / (width * 0.4)) * (height * 0.08)
            points.append((x, y))
        for x in np.linspace(width * 0.3, width * 0.7, 15):
            y = height * 0.31 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.08)
            points.append((x, y))
        for x in np.linspace(width * 0.7, width * 0.3, 15):
            y = height * 0.39 + ((width * 0.7 - x) / (width * 0.4)) * (height * 0.08)
            points.append((x, y))
        for x in np.linspace(width * 0.3, width * 0.7, 20):
            y = height * 0.47 + ((x - width * 0.3) / (width * 0.4)) * (height * 0.33)
            points.append((x, y))
    
    return [(p[0] * scale + (width - 400 * scale) / 2, p[1] * scale + (height - 400 * scale) / 2) for p in points]

def count_fingers(hand_landmarks):
    fingers = 0
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if thumb_tip.x > thumb_ip.x:
        fingers += 1
    
    for tip_id in [8, 12, 16, 20]:
        tip = landmarks[tip_id]
        pip = landmarks[tip_id - 2]
        if tip.y < pip.y:
            fingers += 1
    
    return fingers

def draw_hand_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    for start, end in connections:
        start_landmark = landmarks[start]
        end_landmark = landmarks[end]
        start_pt = (int(start_landmark.x * w), int(start_landmark.y * h))
        end_pt = (int(end_landmark.x * w), int(end_landmark.y * h))
        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

def main():
    base_options = python.BaseOptions(model_asset_path='/Users/tanushpatel/.cache/mediapipe/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hands = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    particles = []
    current_number = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hands.detect(mp_image)
        
        detected_number = 0
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
                draw_hand_landmarks(frame, landmarks)
                detected_number = count_fingers(landmarks)
        
        if detected_number != current_number:
            current_number = detected_number
            frame_count = 0
        
        if current_number > 0 and frame_count % 3 == 0:
            points = get_number_points(current_number, frame.shape[1], frame.shape[0])
            for _ in range(3):
                point = random.choice(points)
                particle = Particle(
                    random.randint(0, frame.shape[1]),
                    random.randint(0, frame.shape[0]),
                    point[0], point[1]
                )
                particles.append(particle)
        
        frame_count += 1
        
        for particle in particles:
            particle.update()
            particle.draw(frame)
        
        particles = [p for p in particles if p.is_alive()]
        
        if current_number > 0:
            points = get_number_points(current_number, frame.shape[1], frame.shape[0])
            for i, point in enumerate(points):
                if i % 3 == 0:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)
        
        cv2.putText(frame, f"Fingers: {current_number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Hand Gesture Particles", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# Hand Gesture Particle Visualization

A real-time hand gesture recognition application that displays colorful particle effects flowing toward the shape of the detected number (based on finger count).

## Features

- **Real-time hand tracking** using MediaPipe
- **Finger counting** (1-5 fingers supported)
- **Particle effects** that flow toward number shapes
- **Hand skeleton visualization** with landmark points
- **Webcam input** at 1280x720 resolution

## Requirements

- Python 3.8+
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
   ```bash
   python hand_particles.py
   ```

2. Show your hand to the webcam
3. Hold up 1-5 fingers to see particles flow toward that number's shape
4. Press `q` to quit

## Controls

| Input | Action |
|-------|--------|
| 1-5 fingers | Display particles flowing to corresponding number |
| `q` | Quit application |

## How It Works

1. MediaPipe HandLandmarker detects hand landmarks in real-time
2. Finger counting algorithm determines how many fingers are raised
3. Particles spawn randomly and gravitate toward points forming the detected number
4. Hand skeleton is drawn with green landmarks and connections
5. Particle alpha fades as they reach their target, creating a trail effect

## Files

- `hand_particles.py` - Main application
- `requirements.txt` - Python dependencies

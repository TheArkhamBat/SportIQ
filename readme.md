# SportIQ

Real-time athletic performance analytics using YOLOv8 pose estimation. Tracks 17 body joints, calculates limb velocities, joint angles, gait symmetry, and injury risk assessment.

## Installation

```bash
# Clone repository
git clone https://github.com/TheArkhamBat/SportIQ.git
cd SportIQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy scipy

# Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```
## Usage

### Live Webcam Analysis
```bash
python main.py
```

Controls:

    q - Quit

    f - Toggle fullscreen

What it shows

    Stick figure - 17 joint skeleton with yellow lines

    Limb velocities - Speed of arms/legs (px/s)

    Joint angles - Elbow and knee angles (degrees)

    Injury risk - Color-coded (green/yellow/red)

    Gait analysis - Step symmetry % and cadence

Video File Analysis
bash

```bash
python analyze_video.py --video path/to/video.mp4
```

Requirements

    Python 3.8 - 3.10

    Webcam (for live mode)

    8GB RAM minimum


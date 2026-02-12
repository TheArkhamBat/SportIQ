# SportIQ - Advanced Athletic Performance Analytics

Python 3.8+ | YOLOv8 Pose | PyTorch | OpenCV | MIT License

Real-time pose estimation, biomechanical analysis, injury risk assessment, and tactical intelligence.

---

## INSTALLATION

### Prerequisites
- Python 3.8 - 3.10
- 8GB RAM minimum
- 5GB storage

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/TheArkhamBat/SportIQ.git
cd SportIQ

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# .\venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install ultralytics opencv-python numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Verify installation
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt'); print('Installation successful')"



# Basic pose detection
python main.py --mode analyze --video path/to/video.mp4

# With athlete ID
python main.py --mode analyze --video data/sample.mp4 --athlete player_01

# Complete analysis with injury report
python video_analysis_complete_fixed.py --video data/running.mp4 --athlete Athlete_Name




# Simple webcam pose detection
python realtime_fix.py

# Full analysis with all metrics
python sportiq_complete.py




# Start REST API
python main.py --mode api

# Endpoints:
# GET  http://localhost:8000/health
# POST http://localhost:8000/analyze/video
# GET  http://localhost:8000/performance/report?athlete_id=player_09




cd data/raw
wget -O test.avi https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi
wget -O running.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4
cd ../..
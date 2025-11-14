# Ai-Face-Video-Extractor
An intelligent desktop application that uses AI-powered face detection to filter videos, keeping only frames containing human faces. Built with Python and OpenCV, this tool leverages advanced computer vision models to automate video processing tasks.



## 🧠 AI Features

### Dual Detection Models
- **SSD (Single Shot MultiBox Detector)**: Deep learning-based detection with confidence scoring
- **Haar Cascade**: Traditional machine learning approach for face detection
- **Configurable Confidence Threshold**: Adjust detection sensitivity from 0.1 to 1.0

### Smart Video Processing
- **Intelligent Frame Selection**: Automatically identifies and retains frames with human faces
- **Real-time Processing**: Live progress tracking with frames-per-second monitoring
- **Audio Preservation**: Maintains synchronized audio from original video
- **Batch Frame Export**: Option to save detected frames as images or separate videos

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- OpenCV with DNN support
- FFmpeg (for audio processing)
- Pre-trained AI models

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/face-video-processor.git
cd face-video-processor
```

2. Install required packages:
```bash
pip install opencv-python tkinter numpy
```

3. Download AI models:
   - SSD Model: `res10_300x300_ssd_iter_140000.caffemodel`
   - Config: `deploy.prototxt`
   - Haar Cascade: `haarcascade_frontalface_default.xml`

4. Install FFmpeg and update the path in the configuration

### Usage
```python
python face_video_processor.py
```

## 🎯 How the AI Works

### Face Detection Pipeline
1. **Frame Extraction**: Video is decomposed into individual frames
2. **AI Analysis**: Each frame is processed through selected detection model
3. **Confidence Filtering**: Frames are kept based on confidence threshold
4. **Temporal Reconstruction**: Selected frames are reassembled into video

### Model Comparison
- **SSD Model**: Higher accuracy, better with varied angles and lighting
- **Haar Cascade**: Faster processing, effective for frontal faces

## ⚙️ Configuration

### Model Paths
Update these paths in the code:
```python
self.model_path = "path/to/res10_300x300_ssd_iter_140000.caffemodel"
self.config_path = "path/to/deploy.prototxt" 
self.cascade_path = "path/to/haarcascade_frontalface_default.xml"
self.ffmpeg_path = "path/to/ffmpeg.exe"
```

### Detection Settings
- **Confidence Threshold**: 0.1 (most sensitive) to 1.0 (most strict)
- **Detection Method**: Choose between SSD (recommended) or Haar Cascade
- **Output Format**: Video, images, or both

## 📊 Performance Metrics

The application provides real-time AI processing statistics:
- Frames processed per second
- Detection confidence scores
- Frame retention percentage
- Estimated time remaining
- Processing speed optimization

## 🎨 Features

### Input/Output
- Support for MP4, AVI, MOV formats
- Custom output naming with AI parameters
- Flexible save locations

### Processing Options
- Adjustable confidence thresholds
- Multiple output formats (video, images, both)
- Audio preservation toggle
- Real-time progress monitoring

### User Interface
- Intuitive GUI with progress tracking
- Detailed processing logs
- One-click output video opening
- Stop/resume functionality

## 🔧 Technical Details

### AI Models Used
1. **SSD with ResNet-10**: Pre-trained on WIDER FACE dataset
2. **Haar Cascade**: OpenCV's classic face detector
3. **OpenCV DNN Module**: For neural network inference

### Processing Pipeline
```
Video Input → Frame Extraction → AI Detection → Confidence Filtering → 
Frame Selection → Video Reconstruction → Audio Sync → Final Output
```

### Supported Resolutions
- 480p to 4K videos
- Various aspect ratios
- Multiple frame rates (24-60 FPS)

## 📈 Use Cases

### Content Creation
- Extract face-containing scenes from long videos
- Create highlight reels from surveillance footage
- Generate training data for ML models

### Research & Development
- Computer vision research
- Face detection algorithm testing
- Video dataset preprocessing

### Security & Monitoring
- Motion-activated recording filtering
- Person-of-interest tracking
- Automated surveillance review

## 🛠️ Customization

### Extending Detection
The modular architecture allows easy integration of:
- Additional detection models (YOLO, MTCNN)
- Custom object classifiers
- Multiple face tracking
- Emotion recognition

### Output Modifications
- Custom video codecs
- Resolution scaling
- Frame rate adjustment
- Watermark addition

## 🤝 Contributing

We welcome contributions to enhance the AI capabilities:
- New detection models
- Performance optimizations
- Additional features
- Bug fixes and improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision libraries
- Contributors to pre-trained AI models
- FFmpeg for audio/video processing
- Python community for excellent tooling

---

**Built with AI for intelligent video processing** 🚀

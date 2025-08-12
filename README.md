# Video Player Application

A simple Python video player with GUI interface that allows users to select and play video files.

## Features

- File picker dialog to select video files
- Video playback with play/pause/stop controls
- Progress bar with seek functionality
- Time display showing current position and total duration
- Support for common video formats (MP4, AVI, MOV, MKV, WMV, FLV, WebM)

## Requirements

- Python 3.6+
- Required packages (install via `pip install -r requirements.txt`):
  - opencv-python
  - pillow
  - tkinter (usually comes with Python)
  - pygame

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python video_player.py
```

3. Click "Select Video File" to choose a video file
4. Use the play/pause/stop buttons to control playback
5. Use the progress bar to seek to different positions in the video

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- WMV
- FLV
- WebM
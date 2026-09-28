# PalmPilot ✋✈️

PalmPilot is a Python-based desktop application that lets you control your computer's scrolling using real-time hand gestures. Built with the latest MediaPipe Tasks API, it provides a smooth, touchless scrolling experience perfect for reading long articles, cooking with messy hands, or just feeling like a wizard.

## ✨ Features

- **3 Gesture Modes**:
  - **Swipe (Default)**: Move your hand up/down in the frame to scroll. (Most natural and reliable)
  - **Pinch**: Pinch your thumb and index finger to scroll down, spread them to scroll up.
  - **Fingers**: Hold up an open hand (4-5 fingers) to scroll up, or a closed fist (0-1 fingers) to scroll down.
- **Anti-Flickering Stabilization**: Uses Exponential Moving Average (EMA) smoothing, gesture confirmation framing, and hysteresis to eliminate jitter and accidental scrolls.
- **Multi-Camera Support**: Automatically detects available cameras on startup. Easily use your smartphone as a high-quality webcam via Phone Link.
- **Live HUD**: Heads-Up Display showing your current mode, camera, and active scroll state.

## 🛠️ Tech Stack

- **Python**
- **OpenCV** (`opencv-python`) - Camera capture and UI rendering
- **MediaPipe** (`mediapipe`) - Blazing fast hand tracking (Tasks API)
- **PyAutoGUI** (`PyAutoGUI`) - System-level scroll simulation

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/No-Thing01/PalmPilot.git
   cd PalmPilot
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MediaPipe Model:**
   Download the `hand_landmarker.task` model file and place it in the same directory as `main.py`:
   [Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

5. **Run the application:**
   ```bash
   python main.py
   ```

## 🎮 Controls

While the application window is active, you can use the following keyboard controls:

- **`M`** : Cycle through gesture modes (SWIPE -> PINCH -> FINGERS)
- **`C`** : Cycle through available cameras (Great for switching to Phone Link)
- **`Q` or `ESC`** : Quit the application

## ⚙️ Customization

You can easily tweak the sensitivity, scroll speed, and stabilization by editing the variables at the top of the `main()` function in `main.py`:

```python
# --- CUSTOMIZABLE VARIABLES ---
PINCH_THRESHOLD = 40         # Pixel distance below this = Scroll Down (pinch mode)
SPREAD_THRESHOLD = 150       # Pixel distance above this = Scroll Up (pinch mode)
SCROLL_SPEED = 20            # Scroll units per frame
SWIPE_THRESHOLD = 0.018      # Normalized Y delta to trigger a swipe scroll
SMOOTHING_ALPHA = 0.4        # Landmark EMA smoothing (0.3=very smooth, 0.7=responsive)
# ------------------------------
```

## 📄 License

MIT License - feel free to modify and use it for your own projects!

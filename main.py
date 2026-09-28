import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections,
    RunningMode, drawing_utils, drawing_styles,
)
import pyautogui
import math
import time
import os
import sys

# Disable PyAutoGUI failsafe and pause to ensure smooth real-time performance
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Resolve model path relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")


# ──────────────────────────────────────────────────────────
#  Camera Utilities
# ──────────────────────────────────────────────────────────

def find_available_cameras(max_check=10):
    """Probe camera indices 0..max_check-1 and return a list of working indices."""
    available = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow is faster on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def select_camera():
    """List available cameras and let the user pick one at startup."""
    print("\nSearching for cameras...")
    cameras = find_available_cameras()

    if not cameras:
        print("ERROR: No cameras found!")
        sys.exit(1)

    if len(cameras) == 1:
        print(f"Found 1 camera (index {cameras[0]}). Using it automatically.")
        return cameras[0]

    print(f"\nFound {len(cameras)} cameras:")
    for cam in cameras:
        label = "Built-in webcam" if cam == 0 else f"Camera {cam} (e.g. Phone Link / USB)"
        print(f"  [{cam}] {label}")

    while True:
        try:
            choice = int(input(f"\nEnter camera index [{cameras[0]}]: ") or cameras[0])
            if choice in cameras:
                return choice
            print(f"Invalid. Choose from: {cameras}")
        except ValueError:
            print("Please enter a number.")


def switch_camera(current_cap, current_idx, available_cameras):
    """Cycle to the next available camera. Returns (new_cap, new_idx)."""
    if len(available_cameras) <= 1:
        return current_cap, current_idx

    # Find the next camera in the list
    try:
        pos = available_cameras.index(current_idx)
    except ValueError:
        pos = -1
    next_idx = available_cameras[(pos + 1) % len(available_cameras)]

    current_cap.release()
    new_cap = cv2.VideoCapture(next_idx, cv2.CAP_DSHOW)
    if not new_cap.isOpened():
        print(f"Failed to open camera {next_idx}, reverting...")
        new_cap = cv2.VideoCapture(current_idx, cv2.CAP_DSHOW)
        return new_cap, current_idx

    print(f"Switched to camera {next_idx}")
    return new_cap, next_idx


# ──────────────────────────────────────────────────────────
#  Smoothing & Stabilization
# ──────────────────────────────────────────────────────────

class LandmarkSmoother:
    """
    Exponential Moving Average (EMA) smoother for hand landmarks.
    Blends each new frame's landmarks with the previous smoothed values,
    eliminating the raw jitter that causes gesture flickering.

    alpha = 1.0 → no smoothing (raw values)
    alpha = 0.3 → heavy smoothing (very stable but laggy)
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None  # list of (x, y, z) tuples for 21 landmarks

    def smooth(self, hand_lms):
        """Takes a list of NormalizedLandmark, returns smoothed (x, y, z) list."""
        current = [(lm.x, lm.y, lm.z) for lm in hand_lms]

        if self.prev is None:
            self.prev = current
            return current

        smoothed = []
        for (cx, cy, cz), (px, py, pz) in zip(current, self.prev):
            sx = self.alpha * cx + (1 - self.alpha) * px
            sy = self.alpha * cy + (1 - self.alpha) * py
            sz = self.alpha * cz + (1 - self.alpha) * pz
            smoothed.append((sx, sy, sz))

        self.prev = smoothed
        return smoothed

    def reset(self):
        self.prev = None


class GestureState:
    """
    Stabilized gesture state machine that prevents flickering by requiring:
    1. CONFIRMATION: N consecutive frames agreeing on a direction before triggering.
    2. COOLDOWN: After triggering, wait M frames before allowing a new trigger.
    3. HYSTERESIS: Uses separate enter/exit thresholds to prevent oscillation.
    """

    def __init__(self, confirm_frames=4, cooldown_frames=6):
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.pending_direction = None
        self.pending_count = 0
        self.cooldown = 0
        self.active_direction = None  # Currently active scroll direction

    def update(self, raw_direction):
        """
        Feed a raw per-frame direction ("UP", "DOWN", or None).
        Returns the confirmed direction to act on, or None.
        """
        # Count down cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            self.active_direction = None
            return None

        # If no gesture detected this frame
        if raw_direction is None:
            self.pending_direction = None
            self.pending_count = 0
            self.active_direction = None
            return None

        # If same direction as pending, increment counter
        if raw_direction == self.pending_direction:
            self.pending_count += 1
        else:
            # New direction — reset counter
            self.pending_direction = raw_direction
            self.pending_count = 1

        # Confirmed after N consecutive frames
        if self.pending_count >= self.confirm_frames:
            self.active_direction = raw_direction
            self.cooldown = self.cooldown_frames
            self.pending_count = 0
            return raw_direction

        return None

    def reset(self):
        self.pending_direction = None
        self.pending_count = 0
        self.cooldown = 0
        self.active_direction = None


# ──────────────────────────────────────────────────────────
#  Gesture Detection Helpers
# ──────────────────────────────────────────────────────────

def detect_pinch_spread(smoothed, w, h, pinch_thresh, spread_thresh,
                        pinch_release=None, spread_release=None, last_direction=None):
    """
    Pinch/Spread gesture with HYSTERESIS.
    - Enters pinch at distance < pinch_thresh, exits at distance > pinch_release.
    - Enters spread at distance > spread_thresh, exits at distance < spread_release.
    This prevents flickering when the distance hovers near a threshold.
    """
    if pinch_release is None:
        pinch_release = pinch_thresh + 20   # Must open past 60 to stop "down"
    if spread_release is None:
        spread_release = spread_thresh - 20  # Must close past 130 to stop "up"

    thumb = smoothed[4]
    index = smoothed[8]
    x1, y1 = int(thumb[0] * w), int(thumb[1] * h)
    x2, y2 = int(index[0] * w), int(index[1] * h)
    dist = math.hypot(x2 - x1, y2 - y1)

    # Hysteresis logic
    if last_direction == "DOWN":
        # Stay in DOWN until distance rises above release threshold
        if dist < pinch_release:
            return "DOWN", dist
        else:
            return None, dist
    elif last_direction == "UP":
        # Stay in UP until distance drops below release threshold
        if dist > spread_release:
            return "UP", dist
        else:
            return None, dist
    else:
        # Not currently in a gesture — check entry thresholds
        if dist < pinch_thresh:
            return "DOWN", dist
        elif dist > spread_thresh:
            return "UP", dist
        return None, dist


def detect_swipe(smoothed, prev_wrist_y, swipe_threshold):
    """
    Swipe gesture using smoothed wrist Y-position.
    """
    wrist_y = smoothed[0][1]  # smoothed (x, y, z) tuple, index 1 = y

    if prev_wrist_y is None:
        return None, wrist_y

    delta = wrist_y - prev_wrist_y

    if delta > swipe_threshold:
        return "DOWN", wrist_y
    elif delta < -swipe_threshold:
        return "UP", wrist_y
    return None, wrist_y


def detect_finger_count(smoothed):
    """
    Counts raised fingers using smoothed coordinates.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    raised = 0

    # Thumb
    wrist_x = smoothed[0][0]
    thumb_tip_x = smoothed[tips[0]][0]
    thumb_pip_x = smoothed[pips[0]][0]
    if abs(thumb_tip_x - wrist_x) > abs(thumb_pip_x - wrist_x):
        raised += 1

    # Other 4 fingers: tip.y < pip.y means raised
    for i in range(1, 5):
        if smoothed[tips[i]][1] < smoothed[pips[i]][1]:
            raised += 1

    return raised


# ──────────────────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────────────────

def main():
    # --- CUSTOMIZABLE VARIABLES ---
    PINCH_THRESHOLD = 40         # Pixel distance below this = Scroll Down (pinch mode)
    SPREAD_THRESHOLD = 150       # Pixel distance above this = Scroll Up (pinch mode)
    SCROLL_SPEED = 20            # Scroll units per frame
    SWIPE_THRESHOLD = 0.018      # Normalized Y delta to trigger a swipe scroll
    SWIPE_SCROLL_MULTIPLIER = 80 # How fast swipe scrolls (multiplied by delta)
    SMOOTHING_ALPHA = 0.4        # Landmark EMA smoothing (0.3=very smooth, 0.7=responsive)
    CONFIRM_FRAMES = 4           # Consecutive frames needed to confirm a gesture
    COOLDOWN_FRAMES = 8          # Frames to wait after a scroll before allowing another
    # ------------------------------

    # Gesture modes the user can cycle through
    GESTURE_MODES = ["SWIPE", "PINCH", "FINGERS"]
    gesture_mode_idx = 0  # Start with SWIPE (most reliable)

    # Camera selection
    available_cameras = find_available_cameras()
    cam_idx = select_camera()

    # Initialize HandLandmarker
    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    last_timestamp_ms = 0

    # Stabilization state
    smoother = LandmarkSmoother(alpha=SMOOTHING_ALPHA)
    gesture_state = GestureState(confirm_frames=CONFIRM_FRAMES, cooldown_frames=COOLDOWN_FRAMES)
    prev_wrist_y = None
    last_pinch_direction = None   # For hysteresis in pinch mode
    no_hand_frames = 0            # Count frames without detection
    NO_HAND_GRACE = 5             # Keep last state for this many frames

    print("\n" + "=" * 50)
    print("  HAND GESTURE SCROLLER")
    print("=" * 50)
    print(f"  Camera : {cam_idx}")
    print(f"  Mode   : {GESTURE_MODES[gesture_mode_idx]}")
    print("-" * 50)
    print("  Controls:")
    print("    Q / ESC    — Quit")
    print("    M          — Cycle gesture mode")
    print("    C          — Cycle camera source")
    print("=" * 50 + "\n")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video. Retrying...")
            time.sleep(0.1)
            continue

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_mode = GESTURE_MODES[gesture_mode_idx]
        status_text = "Neutral"
        indicator_color = (255, 255, 0)  # Cyan
        raw_direction = None

        if result.hand_landmarks:
            no_hand_frames = 0

            for hand_lms in result.hand_landmarks:
                # Draw hand skeleton (uses raw landmarks for accurate overlay)
                drawing_utils.draw_landmarks(
                    img, hand_lms,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                # Smooth the landmarks to eliminate jitter
                smoothed = smoother.smooth(hand_lms)

                # ── SWIPE MODE ──
                if current_mode == "SWIPE":
                    direction, prev_wrist_y = detect_swipe(
                        smoothed, prev_wrist_y, SWIPE_THRESHOLD
                    )
                    # Visualize smoothed wrist tracking point
                    wx, wy = int(smoothed[0][0] * w), int(smoothed[0][1] * h)
                    cv2.circle(img, (wx, wy), 14, (0, 165, 255), cv2.FILLED)
                    cv2.putText(img, "Wrist", (wx + 18, wy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    raw_direction = direction

                # ── PINCH MODE ──
                elif current_mode == "PINCH":
                    direction, dist = detect_pinch_spread(
                        smoothed, w, h, PINCH_THRESHOLD, SPREAD_THRESHOLD,
                        last_direction=last_pinch_direction,
                    )
                    last_pinch_direction = direction
                    # Visualize thumb-index connection using smoothed coords
                    x1, y1 = int(smoothed[4][0] * w), int(smoothed[4][1] * h)
                    x2, y2 = int(smoothed[8][0] * w), int(smoothed[8][1] * h)
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.putText(img, f"Dist: {int(dist)}", (cx + 20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    raw_direction = direction

                # ── FINGERS MODE ──
                elif current_mode == "FINGERS":
                    count = detect_finger_count(smoothed)
                    # Visualize finger count
                    wx, wy = int(smoothed[0][0] * w), int(smoothed[0][1] * h)
                    cv2.putText(img, f"Fingers: {count}", (wx - 30, wy + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    if count >= 4:
                        raw_direction = "UP"
                    elif count <= 1:
                        raw_direction = "DOWN"

            # Feed raw direction through the gesture state machine
            confirmed = gesture_state.update(raw_direction)
            if confirmed == "DOWN":
                pyautogui.scroll(-SCROLL_SPEED)
            elif confirmed == "UP":
                pyautogui.scroll(SCROLL_SPEED)

        else:
            # No hand detected
            no_hand_frames += 1
            if no_hand_frames > NO_HAND_GRACE:
                # Only reset after grace period to avoid flicker on brief dropouts
                smoother.reset()
                gesture_state.reset()
                prev_wrist_y = None
                last_pinch_direction = None

        # Determine display status from gesture state
        active = gesture_state.active_direction
        if active == "DOWN":
            status_text = "Scrolling DOWN"
            indicator_color = (0, 0, 255)
        elif active == "UP":
            status_text = "Scrolling UP"
            indicator_color = (0, 255, 0)

        # ── HUD (Heads-Up Display) ──
        cv2.putText(img, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, indicator_color, 3)
        mode_label = f"Mode: {current_mode}  [M]"
        cv2.putText(img, mode_label, (w - 280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cam_label = f"Cam: {cam_idx}  [C] switch  |  [Q/ESC] quit"
        cv2.putText(img, cam_label, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Hand Gesture Scroller", img)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('m') or key == ord('M'):
            gesture_mode_idx = (gesture_mode_idx + 1) % len(GESTURE_MODES)
            smoother.reset()
            gesture_state.reset()
            prev_wrist_y = None
            last_pinch_direction = None
            print(f"Gesture mode: {GESTURE_MODES[gesture_mode_idx]}")
        elif key == ord('c') or key == ord('C'):
            cap, cam_idx = switch_camera(cap, cam_idx, available_cameras)
            last_timestamp_ms = 0
            smoother.reset()

    # Clean up
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


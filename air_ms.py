import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
import urllib.request
from collections import deque

# Configure PyAutoGUI for maximum smoothness
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Get screen dimensions
screen_w, screen_h = pyautogui.size()
print(f"Screen resolution: {screen_w}x{screen_h}")

# Download model if needed
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("\nDownloading hand detection model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit()

# Initialize MediaPipe Hand Landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

detector = vision.HandLandmarker.create_from_options(options)

# Initialize webcam with optimized settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read from webcam")
    exit()

cam_h, cam_w = test_frame.shape[:2]
print(f"Camera resolution: {cam_w}x{cam_h}")

# Landmark indices
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17
THUMB_TIP = 4
WRIST = 0

# Hand connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ENHANCED SMOOTHING SYSTEM
position_buffer = deque(maxlen=8)  # Increased buffer for smoother movement
velocity_buffer = deque(maxlen=3)  # Track velocity for predictive smoothing

# Kalman-like filter parameters
smoothing_factor = 0.7  # Higher = smoother but slightly laggy
velocity_smoothing = 0.6

# Dead zone to reduce micro-jitter
dead_zone = 3  # pixels

# Click detection with improved debouncing
last_click_time = 0
last_right_click_time = 0
click_debounce = 0.25  # Reduced for faster clicking
double_click_window = 0.4  # Time window for double-click
click_count = 0
last_click_release_time = 0
is_dragging = False
drag_start_time = 0
pinch_held = False
pinch_debounce_timer = 0

# Scroll variables
scroll_mode_active = False
last_scroll_y = None
scroll_sensitivity = 3
scroll_buffer = deque(maxlen=3)

# Speed zones - move slower when near edges for precision
edge_zone = 100  # pixels from screen edge
edge_speed_factor = 0.6

# Pause state
paused = False
frame_count = 0
show_tips = True

print("\n" + "="*70)
print("🖱️  ULTRA-SMOOTH AIR MOUSE - OPTIMIZED EDITION")
print("="*70)
print("\n✋ GESTURES:")
print("   Index finger up          -> Move cursor (MAIN MODE)")
print("   Close fist (all down)    -> LEFT CLICK")
print("   Two quick fists          -> DOUBLE CLICK")
print("   Middle finger up only    -> RIGHT CLICK")
print("   Hold fist (0.7s)         -> START DRAG (open hand to end)")
print("   All 4 fingers up         -> SCROLL MODE")
print("\n⌨️  KEYBOARD CONTROLS:")
print("   ESC or Q  → Quit")
print("   P         → Pause/Resume")
print("   R         → Reset tracking")
print("   T         → Toggle tips display")
print("   + / -     → Adjust smoothing")
print("\n💡 PRO TIPS FOR BEST EXPERIENCE:")
print("   1. Keep hand 1.5-2 feet from camera")
print("   2. Ensure good lighting (front or side light)")
print("   3. Keep hand steady - move arm, not just fingers")
print("   4. Make deliberate gestures, avoid rapid movements")
print("   5. Use your whole palm as a 'base' for stability")
print("   6. For precision: move slower near targets")
print("   7. Practice smooth, flowing motions")
print("\nCLICKING TECHNIQUES:")
print("   LEFT CLICK: Point with index, then CLOSE FIST quickly")
print("   DOUBLE CLICK: Two quick fists (within 0.4 seconds)")
print("   DRAG: Close fist and hold for 0.7 seconds, then move")
print("   RIGHT CLICK: Middle finger up only (others down)")
print("\nGESTURE TIPS:")
print("   - For cursor: Only index finger extended")
print("   - For click: Make a quick fist motion")
print("   - For right-click: Only middle finger sticks up")
print("   - For drag: Hold fist steady before moving")
print("\n💡 PRO TIPS FOR BEST EXPERIENCE:")
print("   1. Keep hand 1.5-2 feet from camera")
print("   2. Ensure good lighting (front or side light)")
print("   3. Keep hand steady - move arm, not just fingers")
print("   4. Make deliberate gestures, avoid rapid movements")
print("   5. Use your whole palm as a 'base' for stability")
print("   6. For precision: move slower near targets")
print("   7. Practice smooth, flowing motions")
print("\n🖱️ CLICKING TECHNIQUES:")
print("   • LEFT CLICK: Bring thumb and index close together, then release")
print("   • DOUBLE CLICK: Do two quick pinches (within 0.4 seconds)")
print("   • DRAG: Pinch and hold for 0.7 seconds, then move")
print("   • RIGHT CLICK: Make peace sign with index + middle fingers")
print("\n⚠️  PINCH TIPS:")
print("   - Don't touch fingers together - just bring them CLOSE")
print("   - Keep other fingers relaxed and slightly curled")
print("   - Make it a quick 'snap' motion for clicks")
print("   - For drag: pinch firmly and pause before moving")
print("="*70 + "\n")

def calculate_distance(p1, p2):
    """Calculate distance between two landmarks"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_finger_extended(tip, mcp, wrist):
    """Check if finger is extended"""
    tip_dist = calculate_distance(tip, wrist)
    mcp_dist = calculate_distance(mcp, wrist)
    return tip_dist > mcp_dist * 1.15

def exponential_smoothing(new_val, old_val, factor):
    """Apply exponential smoothing"""
    return old_val + (new_val - old_val) * (1 - factor)

def apply_dead_zone(new_x, new_y, old_x, old_y, zone):
    """Prevent micro-jitter with dead zone"""
    if abs(new_x - old_x) < zone and abs(new_y - old_y) < zone:
        return old_x, old_y
    return new_x, new_y

def get_ultra_smoothed_position(x, y, prev_x, prev_y):
    """Advanced smoothing with velocity prediction"""
    # Add to buffer
    position_buffer.append((x, y))
    
    if len(position_buffer) < 2:
        return x, y
    
    # Calculate weighted moving average (recent positions have more weight)
    weights = np.linspace(0.5, 1.0, len(position_buffer))
    weights = weights / weights.sum()
    
    avg_x = sum(p[0] * w for p, w in zip(position_buffer, weights))
    avg_y = sum(p[1] * w for p, w in zip(position_buffer, weights))
    
    # Apply exponential smoothing
    if prev_x > 0:
        smooth_x = exponential_smoothing(avg_x, prev_x, smoothing_factor)
        smooth_y = exponential_smoothing(avg_y, prev_y, smoothing_factor)
    else:
        smooth_x, smooth_y = avg_x, avg_y
    
    # Apply dead zone
    smooth_x, smooth_y = apply_dead_zone(smooth_x, smooth_y, prev_x, prev_y, dead_zone)
    
    return int(smooth_x), int(smooth_y)

def apply_edge_slowdown(x, y, screen_x, screen_y):
    """Slow down cursor near screen edges for precision"""
    # Check if near edges
    near_left = screen_x < edge_zone
    near_right = screen_x > screen_w - edge_zone
    near_top = screen_y < edge_zone
    near_bottom = screen_y > screen_h - edge_zone
    
    if near_left or near_right or near_top or near_bottom:
        return x, y, True
    return x, y, False

# Previous positions for smoothing
prev_smooth_x, prev_smooth_y = 0, 0
prev_screen_x, prev_screen_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    try:
        results = detector.detect_for_video(mp_image, frame_count)
    except Exception as e:
        continue
    
    mode_text = "🔍 Searching for hand..."
    mode_color = (100, 100, 100)
    
    if paused:
        mode_text = "⏸️  PAUSED - Press 'P' to resume"
        mode_color = (0, 165, 255)
    
    elif results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw hand skeleton (thinner for less distraction)
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                start_point = (int(start.x * cam_w), int(start.y * cam_h))
                end_point = (int(end.x * cam_w), int(end.y * cam_h))
                
                cv2.line(frame, start_point, end_point, (0, 200, 0), 1)
            
            # Draw landmarks (smaller circles)
            for idx, landmark in enumerate(hand_landmarks):
                x, y = int(landmark.x * cam_w), int(landmark.y * cam_h)
                if idx in [INDEX_TIP, THUMB_TIP]:
                    cv2.circle(frame, (x, y), 6, (255, 100, 255), -1)
                else:
                    cv2.circle(frame, (x, y), 3, (100, 255, 255), -1)
            
            # Get landmarks
            index_tip = hand_landmarks[INDEX_TIP]
            index_mcp = hand_landmarks[INDEX_MCP]
            middle_tip = hand_landmarks[MIDDLE_TIP]
            middle_mcp = hand_landmarks[MIDDLE_MCP]
            ring_tip = hand_landmarks[RING_TIP]
            ring_mcp = hand_landmarks[RING_MCP]
            pinky_tip = hand_landmarks[PINKY_TIP]
            pinky_mcp = hand_landmarks[PINKY_MCP]
            thumb_tip = hand_landmarks[THUMB_TIP]
            wrist = hand_landmarks[WRIST]
            
            # Check which fingers are extended
            index_extended = is_finger_extended(index_tip, index_mcp, wrist)
            middle_extended = is_finger_extended(middle_tip, middle_mcp, wrist)
            ring_extended = is_finger_extended(ring_tip, ring_mcp, wrist)
            pinky_extended = is_finger_extended(pinky_tip, pinky_mcp, wrist)
            
            extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
            
            # Detect fist (all fingers down)
            is_fist = extended_count == 0
            
            # Track fist state for click detection
            if is_fist and not pinch_held:
                pinch_held = True
                drag_start_time = current_time
            elif not is_fist and pinch_held:
                # Fist released - trigger click or end drag
                pinch_held = False
                
                if is_dragging:
                    # End drag
                    pyautogui.mouseUp()
                    is_dragging = False
                    mode_text = "Drag released"
                    mode_color = (0, 255, 0)
                    click_count = 0
                    last_click_release_time = current_time
                else:
                    # Check for double-click
                    if current_time - last_click_release_time < double_click_window:
                        pyautogui.doubleClick()
                        mode_text = "DOUBLE CLICK!"
                        mode_color = (255, 100, 255)
                        click_count = 0
                    else:
                        # Single click
                        pyautogui.click()
                        mode_text = "LEFT CLICK"
                        mode_color = (0, 255, 0)
                        click_count = 1
                    
                    last_click_release_time = current_time
                    last_click_time = current_time
                
                drag_start_time = 0
            
            # Calculate pinch distance (still used for visual feedback)
            pinch_dist = calculate_distance(thumb_tip, index_tip)
            is_pinching = pinch_dist < 0.08
            
            # Get cursor position (use index finger tip)
            cursor_x = int(index_tip.x * cam_w)
            cursor_y = int(index_tip.y * cam_h)
            
            # Apply ultra-smooth filtering
            smooth_x, smooth_y = get_ultra_smoothed_position(cursor_x, cursor_y, prev_smooth_x, prev_smooth_y)
            prev_smooth_x, prev_smooth_y = smooth_x, smooth_y
            
            # Map to screen coordinates
            screen_x = np.interp(smooth_x, [0, cam_w], [0, screen_w])
            screen_y = np.interp(smooth_y, [0, cam_h], [0, screen_h])
            
            # Apply edge slowdown if near borders
            screen_x, screen_y, near_edge = apply_edge_slowdown(smooth_x, smooth_y, screen_x, screen_y)
            
            if near_edge:
                # Blend with previous position for extra smoothness at edges
                screen_x = prev_screen_x + (screen_x - prev_screen_x) * edge_speed_factor
                screen_y = prev_screen_y + (screen_y - prev_screen_y) * edge_speed_factor
            
            prev_screen_x, prev_screen_y = screen_x, screen_y
            
            # Move cursor
            if not paused:
                pyautogui.moveTo(int(screen_x), int(screen_y))
            
            current_time = time.time()
            
            # GESTURE RECOGNITION
            
            # All fingers extended - SCROLL MODE
            if extended_count >= 4:
                mode_text = "SCROLL MODE - Move hand up/down"
                mode_color = (255, 165, 0)
                scroll_mode_active = True
                
                if last_scroll_y is not None:
                    scroll_delta = last_scroll_y - smooth_y
                    scroll_buffer.append(scroll_delta)
                    
                    # Average scroll delta for smoothness
                    if len(scroll_buffer) >= 2:
                        avg_delta = sum(scroll_buffer) / len(scroll_buffer)
                        if abs(avg_delta) > 8:
                            scroll_amount = int(avg_delta / 8) * scroll_sensitivity
                            pyautogui.scroll(scroll_amount)
                
                last_scroll_y = smooth_y
                
                # Reset drag states
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
            
            # Fist held - CLICK or DRAG
            elif is_fist:
                hold_duration = current_time - drag_start_time if drag_start_time > 0 else 0
                
                if hold_duration > 0.7:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    mode_text = "DRAGGING - Open hand to drop"
                    mode_color = (255, 255, 0)
                else:
                    mode_text = f"FIST - Release to click"
                    mode_color = (255, 200, 0)
                
                scroll_mode_active = False
                last_scroll_y = None
                scroll_buffer.clear()
            
            # Middle finger only - RIGHT CLICK
            elif not index_extended and middle_extended and not ring_extended and not pinky_extended:
                if current_time - last_right_click_time > click_debounce:
                    pyautogui.rightClick()
                    mode_text = "RIGHT CLICK"
                    mode_color = (255, 0, 255)
                    last_right_click_time = current_time
                else:
                    mode_text = "Middle finger ready"
                    mode_color = (200, 0, 200)
                
                scroll_mode_active = False
                last_scroll_y = None
                scroll_buffer.clear()
            
            # Index finger only - NORMAL CURSOR
            elif index_extended and not middle_extended:
                mode_text = "Moving cursor" + (" [PRECISION MODE]" if near_edge else "")
                mode_color = (0, 255, 255)
                scroll_mode_active = False
                last_scroll_y = None
                scroll_buffer.clear()
            
            # Draw cursor indicator with glow effect
            cv2.circle(frame, (smooth_x, smooth_y), 25, mode_color, 1)
            cv2.circle(frame, (smooth_x, smooth_y), 15, mode_color, 2)
            cv2.circle(frame, (smooth_x, smooth_y), 5, mode_color, -1)
            
            # Draw fist indicator
            if is_fist:
                cv2.circle(frame, (smooth_x, smooth_y), 35, (255, 0, 0), 3)
    
    else:
        # Reset when no hand detected
        position_buffer.clear()
        velocity_buffer.clear()
        scroll_buffer.clear()
        last_scroll_y = None
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
        drag_start_time = 0
        pinch_held = False
        prev_smooth_x, prev_smooth_y = 0, 0
    
    # Display status bar
    cv2.rectangle(frame, (0, cam_h - 60), (cam_w, cam_h), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (10, cam_h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    # Display smoothing level
    smoothing_text = f"Smoothing: {smoothing_factor:.1f} | FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
    cv2.putText(frame, smoothing_text, (10, cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Display controls
    cv2.putText(frame, "ESC/Q: Quit | P: Pause | R: Reset | +/-: Smoothing | T: Tips", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Show tips overlay if enabled
    if show_tips:
        tip_y = 50
        tips = [
            "Keep hand steady",
            "Move arm, not fingers",
            "1.5-2 feet distance"
        ]
        for tip in tips:
            cv2.putText(frame, f"💡 {tip}", (cam_w - 250, tip_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            tip_y += 20
    
    cv2.imshow("Ultra-Smooth Air Mouse", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('p'):
        paused = not paused
        print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    elif key == ord('r'):
        position_buffer.clear()
        velocity_buffer.clear()
        scroll_buffer.clear()
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
        prev_smooth_x, prev_smooth_y = 0, 0
        print("🔄 Tracking reset")
    elif key == ord('t'):
        show_tips = not show_tips
        print(f"Tips display: {'ON' if show_tips else 'OFF'}")
    elif key == ord('+') or key == ord('='):
        smoothing_factor = min(0.9, smoothing_factor + 0.1)
        print(f"Smoothing increased to {smoothing_factor:.1f}")
    elif key == ord('-') or key == ord('_'):
        smoothing_factor = max(0.3, smoothing_factor - 0.1)
        print(f"Smoothing decreased to {smoothing_factor:.1f}")

# Cleanup
if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()
print("\n" + "="*70)
print("✨ Air Mouse closed. Goodbye!")
print("="*70)
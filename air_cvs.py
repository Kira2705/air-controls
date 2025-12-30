import cv2
import numpy as np
import os
import urllib.request

try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
except Exception as e:
    print(f"Error importing mediapipe: {e}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual dimensions
ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read from webcam")
    exit()

h, w = test_frame.shape[:2]
print(f"Webcam resolution: {w}x{h}")

# Download model if needed
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("\nDownloading hand detection model (this may take a minute)...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit()

# Create hand landmarker with LOWER thresholds for better detection
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,  # Lowered from 0.5
    min_hand_presence_confidence=0.3,   # Lowered from 0.5
    min_tracking_confidence=0.3         # Lowered from 0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# Canvas setup
canvas = np.zeros((h, w, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0

# Color palette
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
color_names = ["Blue", "Green", "Red", "Yellow", "Magenta", "Cyan"]
current_color = colors[2]  # Start with Red (more visible)
brush_size = 8  # Bigger default brush

# Hand landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

print("\n" + "="*60)
print("🎨 AIR CANVAS - DEBUG MODE")
print("="*60)
print("\n📌 IMPORTANT TIPS:")
print("   • Hold your hand 1-2 feet from camera")
print("   • Make sure room is well-lit")
print("   • Keep your hand open and fingers visible")
print("\n✋ GESTURES (Watch the debug info on screen):")
print("   ✌️  TWO FINGERS UP (Index + Middle) = DRAW")
print("   👆 ONE FINGER UP (Index only) = CURSOR")
print("\n⌨️  KEYBOARD:")
print("   1-6   → Colors  |  +/-  → Brush size")
print("   c     → Clear   |  s    → Save")
print("   q     → Quit")
print("="*60 + "\n")

frame_count = 0

# Connection pairs for drawing hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    try:
        detection_result = detector.detect_for_video(mp_image, frame_count)
    except Exception as e:
        print(f"Detection error: {e}")
        continue
    
    # Draw color palette
    for i, (color, name) in enumerate(zip(colors, color_names)):
        x_pos = 10 + i * 100
        cv2.rectangle(frame, (x_pos, 10), (x_pos + 90, 50), color, -1)
        if color == current_color:
            cv2.rectangle(frame, (x_pos, 10), (x_pos + 90, 50), (255, 255, 255), 4)
        cv2.putText(frame, name, (x_pos + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display brush size
    cv2.rectangle(frame, (w - 180, 10), (w - 10, 50), (50, 50, 50), -1)
    cv2.putText(frame, f"Brush: {brush_size}px", (w - 170, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Debug info area
    debug_y = 70
    cv2.rectangle(frame, (10, debug_y), (400, debug_y + 120), (0, 0, 0), -1)
    
    mode_text = "🔍 Searching for hand..."
    mode_color = (100, 100, 100)
    finger_status = ""
    
    if detection_result.hand_landmarks:
        hand_detected = True
        cv2.putText(frame, "✓ Hand Detected!", (15, debug_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw hand skeleton (thicker lines)
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
            
            # Draw landmarks (bigger circles)
            for idx, landmark in enumerate(hand_landmarks):
                x, y = int(landmark.x * w), int(landmark.y * h)
                if idx in [INDEX_TIP, MIDDLE_TIP]:
                    cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
                    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            
            # Get key landmarks
            index_tip = hand_landmarks[INDEX_TIP]
            middle_tip = hand_landmarks[MIDDLE_TIP]
            ring_tip = hand_landmarks[RING_TIP]
            pinky_tip = hand_landmarks[PINKY_TIP]
            wrist = hand_landmarks[WRIST]
            
            index_pip = hand_landmarks[INDEX_PIP]
            middle_pip = hand_landmarks[MIDDLE_PIP]
            ring_pip = hand_landmarks[RING_PIP]
            pinky_pip = hand_landmarks[PINKY_PIP]
            
            # Calculate if fingers are extended (comparing to wrist)
            # A finger is "up" if its tip is significantly higher than its PIP joint
            index_up = (index_pip.y - index_tip.y) > 0.03
            middle_up = (middle_pip.y - middle_tip.y) > 0.03
            ring_up = (ring_pip.y - ring_tip.y) > 0.03
            pinky_up = (pinky_pip.y - pinky_tip.y) > 0.03
            
            # Debug: Show finger status
            finger_status = f"Index: {'UP' if index_up else 'down'}  Middle: {'UP' if middle_up else 'down'}"
            finger_status += f"\nRing: {'UP' if ring_up else 'down'}  Pinky: {'UP' if pinky_up else 'down'}"
            
            cv2.putText(frame, f"Index: {'✓ UP' if index_up else '✗ down'}", (15, debug_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if index_up else (100, 100, 100), 2)
            cv2.putText(frame, f"Middle: {'✓ UP' if middle_up else '✗ down'}", (15, debug_y + 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if middle_up else (100, 100, 100), 2)
            cv2.putText(frame, f"Ring: {'✓ UP' if ring_up else '✗ down'}", (15, debug_y + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ring_up else (100, 100, 100), 2)
            
            # Convert to pixel coordinates
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            
            # SIMPLIFIED DRAWING LOGIC - Just two fingers up
            fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])
            
            if index_up and middle_up:
                # DRAW MODE - Use midpoint between index and middle
                mode_text = "✏️ DRAWING MODE - ACTIVE!"
                mode_color = (0, 255, 0)
                
                draw_x = (ix + mx) // 2
                draw_y = (iy + my) // 2
                
                # Show drawing cursor
                cv2.circle(frame, (draw_x, draw_y), brush_size + 5, (255, 255, 255), 3)
                cv2.circle(frame, (draw_x, draw_y), brush_size, current_color, -1)
                
                # Draw on canvas
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (draw_x, draw_y), current_color, brush_size * 2)
                
                prev_x, prev_y = draw_x, draw_y
                
            elif index_up and not middle_up:
                # CURSOR MODE
                mode_text = "👆 Cursor Mode (not drawing)"
                mode_color = (255, 255, 0)
                cv2.circle(frame, (ix, iy), 20, (0, 255, 255), 3)
                cv2.circle(frame, (ix, iy), 5, (0, 255, 255), -1)
                prev_x, prev_y = 0, 0
            else:
                mode_text = "✋ Ready - Show 2 fingers to draw"
                mode_color = (200, 200, 200)
                prev_x, prev_y = 0, 0
    else:
        cv2.putText(frame, "✗ No hand detected", (15, debug_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Show your hand to camera", (15, debug_y + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        prev_x, prev_y = 0, 0
    
    # Display mode at bottom with large background
    text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (5, h - 60), (text_size[0] + 20, h - 5), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 3)
    
    # Merge canvas with frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    result_frame = cv2.add(frame_bg, canvas_fg)
    
    cv2.imshow("Air Canvas - Debug Mode", result_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        print("✓ Canvas cleared!")
    elif key == ord('1'):
        current_color = colors[0]
        print(f"✓ Color: {color_names[0]}")
    elif key == ord('2'):
        current_color = colors[1]
        print(f"✓ Color: {color_names[1]}")
    elif key == ord('3'):
        current_color = colors[2]
        print(f"✓ Color: {color_names[2]}")
    elif key == ord('4'):
        current_color = colors[3]
        print(f"✓ Color: {color_names[3]}")
    elif key == ord('5'):
        current_color = colors[4]
        print(f"✓ Color: {color_names[4]}")
    elif key == ord('6'):
        current_color = colors[5]
        print(f"✓ Color: {color_names[5]}")
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 30)
        print(f"✓ Brush size: {brush_size}px")
    elif key == ord('-') or key == ord('_'):
        brush_size = max(brush_size - 2, 3)
        print(f"✓ Brush size: {brush_size}px")
    elif key == ord('s'):
        filename = f"air_canvas_{np.random.randint(1000, 9999)}.png"
        cv2.imwrite(filename, canvas)
        print(f"✓ Canvas saved as {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n" + "="*60)
print("Thanks for using Air Canvas!")
print("="*60)
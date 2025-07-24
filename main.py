# ================== Gesture Controlled Robot in CoppeliaSim ==================
# This script connects to the CoppeliaSim simulation and controls a robot
# (Pioneer_p3dx) using hand gestures captured in real-time using MediaPipe.
# Gestures:
#   - 0 fingers: move forward
#   - 1 finger: move backward
#   - 2 fingers: turn
#   - 5 fingers: stop
# ==============================================================================

# --- Import required libraries ---
import cv2                    # For video capture and display
import mediapipe as mp        # For hand landmark detection
import sim                    # For CoppeliaSim API
import sys, time              # For system control and delays

# --- Connect to CoppeliaSim simulation ---
print("Connecting to CoppeliaSim...")
sim.simxFinish(-1)  # Close any previous connections
client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if client_id == -1:
    print("❌ Could not connect to CoppeliaSim.")
    sys.exit()
print("✅ Connected!")

# --- Retrieve motor handles from CoppeliaSim ---
_, left_motor = sim.simxGetObjectHandle(client_id, 'Pioneer_p3dx_leftMotor', sim.simx_opmode_blocking)
_, right_motor = sim.simxGetObjectHandle(client_id, 'Pioneer_p3dx_rightMotor', sim.simx_opmode_blocking)

# --- Setup MediaPipe for hand tracking ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

cap = cv2.VideoCapture(0)  # Open webcam
current_command = "STOP"

# --- Main loop for video processing and robot control ---
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB and process hand landmarks
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for id, lm in enumerate(hand_landmark.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        # --- Gesture interpretation based on raised fingers ---
        if lm_list:
            fingers = []
            # Thumb
            fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
            # Other 4 fingers
            for i in range(1, 5):
                fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2] else 0)

            total_fingers = fingers.count(1)

            # Determine command from number of raised fingers
            if total_fingers == 0:
                current_command = "FORWARD"
            elif total_fingers == 1:
                current_command = "REVERSE"
            elif total_fingers == 2:
                current_command = "TURN"
            elif total_fingers == 5:
                current_command = "STOP"

        # --- Send control command to CoppeliaSim ---
        if current_command == "FORWARD":
            sim.simxSetJointTargetVelocity(client_id, left_motor, 2.0, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(client_id, right_motor, 2.0, sim.simx_opmode_oneshot)
        elif current_command == "REVERSE":
            sim.simxSetJointTargetVelocity(client_id, left_motor, -2.0, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(client_id, right_motor, -2.0, sim.simx_opmode_oneshot)
        elif current_command == "TURN":
            sim.simxSetJointTargetVelocity(client_id, left_motor, -1.0, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(client_id, right_motor, 1.0, sim.simx_opmode_oneshot)
        elif current_command == "STOP":
            sim.simxSetJointTargetVelocity(client_id, left_motor, 0.0, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(client_id, right_motor, 0.0, sim.simx_opmode_oneshot)

        # Show command and video frame
        cv2.putText(frame, f"Command: {current_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
sim.simxFinish(client_id)

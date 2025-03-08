import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

data = []
labels = []

print("Show each ASL letter for 3-5 seconds while looking at the camera.")
print("Press the corresponding key on the keyboard for the letter you are signing.")
print("Press 'q' to quit and save the data.")

current_letter = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark data
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Check for keypress to assign a label
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # A key was pressed
                if key == ord('q'):  # Quit and save data
                    break
                elif key >= ord('a') and key <= ord('z'):  # A valid letter key was pressed
                    current_letter = chr(key)
                    print(f"Recording letter: {current_letter}")
                    data.append(landmark_array)
                    labels.append(current_letter)



    # Display the frame
    cv2.imshow("ASL Data Collection", frame)

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save collected data
if data:
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv("asl_data.csv", index=False)
    print(f"Data saved to asl_data.csv with {len(data)} samples.")
else:
    print("No data collected.")
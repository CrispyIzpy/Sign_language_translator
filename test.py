import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import joblib
import time
from transformers import pipeline  # Import Hugging Face pipeline

# Load a pre-trained text generation model from Hugging Face
text_generator = pipeline("text-generation", model="gpt2")

class SignLanguageTranslatorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Create a canvas to display the video feed
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Label to display translated text (as if typing)
        self.translated_text = tk.StringVar()
        self.label = tk.Label(window, textvariable=self.translated_text, font=("Arial", 20))
        self.label.pack()

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.window.destroy()
            return

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {e}")
            self.window.destroy()
            return

        self.mp_drawing = mp.solutions.drawing_utils

        # Load Pre-trained ASL Model (Optional)
        try:
            self.model = joblib.load("asl_model.pkl")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        self.detected_text = ""  # Store translated letters
        self.last_detected_letter = None  # Track the last detected letter
        self.last_detection_time = time.time()  # Track the last detection time
        self.detection_delay = 3  # Delay in seconds between printing letters

        # Start the video loop
        self.update()
        self.window.mainloop()

    def update(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Convert frame for Tkinter display
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Translate hand gestures
                detected_letter = self.translate_gesture(results)
                if detected_letter:
                    current_time = time.time()
                    # Check if the detected letter is new and if the delay has passed
                    if detected_letter != self.last_detected_letter or (current_time - self.last_detection_time) >= self.detection_delay:
                        self.detected_text += detected_letter  # Append letter like typing
                        self.translated_text.set(self.detected_text)
                        self.last_detected_letter = detected_letter
                        self.last_detection_time = current_time

                        # Use Hugging Face Transformers to interpret the detected text
                        self.interpret_with_hugging_face(self.detected_text)

            self.window.after(10, self.update)
        except Exception as e:
            print(f"Error in update loop: {e}")

    def translate_gesture(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                if self.model:
                    prediction = self.model.predict([landmark_array])[0]
                    return prediction  # Return predicted letter
                else:
                    return None
        return None

    def interpret_with_hugging_face(self, text):
        """
        Use Hugging Face Transformers to interpret or improve the detected text.
        """
        try:
            # Generate a response using the Hugging Face model
            response = text_generator(
                f"The following text was detected from sign language gestures: '{text}'. Please correct any errors and provide a meaningful interpretation:",
                max_length=50,
                num_return_sequences=1,
                temperature=0.7
            )

            # Extract the generated text
            llm_output = response[0]['generated_text'].strip()
            print(f"Hugging Face Interpretation: {llm_output}")

            # Update the translated text with the Hugging Face interpretation
            self.translated_text.set(llm_output)
        except Exception as e:
            print(f"Error calling Hugging Face model: {e}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Run Tkinter App
root = tk.Tk()
app = SignLanguageTranslatorApp(root, "Sign Language Translator")
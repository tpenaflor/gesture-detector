import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from data_collector import create_csv, save_landmarks_to_csv
from train_gesture_classifier import train_gesture_classifier

class GestureRecognizer:
    def __init__(self):
        self._init_mediapipe()
        self._init_camera()
        self._load_model()
        self._init_gesture_mapping()
        self.collect_mode = False
        self.current_gesture = None

    def _init_mediapipe(self):
        """Initialize MediaPipe hands solution"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def _init_camera(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _load_model(self):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load('gesture_classifier.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.classes = np.load('class_names.npy', allow_pickle=True)
            print(f"Loaded model for classes: {self.classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None

    def _init_gesture_mapping(self):
        """Initialize gesture key mappings"""
        self.gesture_mapping = {
            ord('1'): 'thumbs_down',
            ord('2'): 'peace',
            ord('3'): 'thumbs_up',
            ord('4'): 'fist'
        }

    def recognize_gestures(self):
        """Main recognition loop"""
        create_csv()
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue

            image = self._process_frame(image)
            results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                self._process_hands(image, results)
            
            self._handle_keypress()
            cv2.imshow('Gesture Recognition', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self._cleanup()

    def _process_frame(self, image):
        """Process and annotate frame"""
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Display mode status
        mode_text = "COLLECT MODE" if self.collect_mode else "RECOGNITION MODE"
        cv2.putText(image, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return image

    def _process_hands(self, image, results):
        """Process and draw hand landmarks"""
        for hand_num, hand_landmarks in enumerate(results.multi_hand_landmarks):
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if self.collect_mode and self.current_gesture:
                self._collect_data(hand_landmarks, hand_num, image)
            elif self.model and not self.collect_mode:
                self._recognize_gesture(hand_landmarks, hand_num, image)

    def _collect_data(self, landmarks, hand_num, image):
        """Handle data collection"""
        save_landmarks_to_csv(self.current_gesture, landmarks)
        cv2.putText(image, f"Saved: {self.current_gesture}", 
                   (10, 70 + hand_num * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _recognize_gesture(self, landmarks, hand_num, image):
        """Perform gesture recognition"""
        gesture = self._detect_gesture(landmarks)
        if gesture:
            cv2.putText(image, gesture, 
                       (10, 70 + hand_num * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _handle_keypress(self):
        """Handle keyboard input"""
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            self._toggle_collection_mode()
        elif key in self.gesture_mapping:
            self.current_gesture = self.gesture_mapping[key]
            print(f"Preparing to collect: {self.current_gesture}")

    def _toggle_collection_mode(self):
        """Toggle between collection and recognition modes"""
        self.collect_mode = not self.collect_mode
        print(f"Collection mode: {self.collect_mode}")
        
        if not self.collect_mode:
            self._retrain_model()

    def _retrain_model(self):
        """Retrain model with new data"""
        train_gesture_classifier()
        self._load_model()  # Reload the updated model

    def _detect_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        try:
            features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            features = features.reshape(1, -1)
            
            if hasattr(self.scaler, 'feature_names_in_'):
                features = pd.DataFrame(features, columns=self.scaler.feature_names_in_)
            
            features_scaled = self.scaler.transform(features)
            gesture = self.model.predict(features_scaled)[0]
            
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(features_scaled))
                return f"{gesture} ({confidence:.2f})"
            return gesture
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.recognize_gestures()
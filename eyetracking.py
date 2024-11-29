import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math

class EyeTrackerMediaPipe:
    def __init__(self):
        self.cap = cv2.VideoCapture(4)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # OSC client setup
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7894)
        
        # Constants for Z calculation
        self.MAX_IPD_PIXELS = 200  # Reduced from 400 - not as close to camera
        self.MIN_IPD_PIXELS = 30   # Reduced from 50 - allow further distance
        
        # Calibration variables
        self.calibrated = False
        self.calibration_pos = None
        self.OFFSET_RANGE = 0.3  # How far from center to edges (adjust as needed)
        
        # Eye state tracking
        self.eyes_were_open = None  # Track previous state
        self.BLINK_THRESHOLD = 0.1  # Adjust this value based on testing
        
    def normalize_coordinates(self, point, frame_width, frame_height):
        """Convert coordinates to 0-1 range relative to calibration point"""
        if not self.calibrated:
            # During calibration, just store the raw normalized position
            return {
                'x': point[0] / frame_width,
                'y': 1 - (point[1] / frame_height)
            }
        
        # Calculate relative position from calibration point
        rel_x = (point[0] / frame_width) - self.calibration_pos['x']
        rel_y = (1 - (point[1] / frame_height)) - self.calibration_pos['y']
        
        # Map to 0-1 range where calibration point is 0.5
        normalized_x = 0.5 + (rel_x / (2 * self.OFFSET_RANGE))
        normalized_y = 0.5 + (rel_y / (2 * self.OFFSET_RANGE))
        
        return {
            'x': max(0, min(1, normalized_x)),
            'y': max(0, min(1, normalized_y))
        }

    def calibrate(self, normalized_pos):
        """Store calibration position"""
        self.calibration_pos = normalized_pos
        self.calibrated = True
        print(f"Calibrated at position: X={normalized_pos['x']:.2f}, Y={normalized_pos['y']:.2f}")

    def calculate_z_value(self, left_eye, right_eye):
        """Calculate normalized Z value based on inter-pupillary distance"""
        if not (left_eye and right_eye):
            return None
            
        # Calculate distance between eyes in pixels
        ipd_pixels = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                            (right_eye[1] - left_eye[1])**2)
        
        # Clamp IPD to our expected range
        ipd_pixels = max(self.MIN_IPD_PIXELS, min(self.MAX_IPD_PIXELS, ipd_pixels))
        
        # Convert to logarithmic scale and normalize to 0-1
        # When close (large IPD) = 0, When far (small IPD) = 1
        z = math.log(ipd_pixels / self.MIN_IPD_PIXELS) / math.log(self.MAX_IPD_PIXELS / self.MIN_IPD_PIXELS)
        z = 1 - max(0, min(1, z))  # Invert and clamp to 0-1
        
        return z

    def send_to_unreal(self, normalized_pos):
        """Send normalized position via OSC"""
        try:
            self.osc_client.send_message("/eyetracking/pos/x", normalized_pos['x'])
            self.osc_client.send_message("/eyetracking/pos/y", normalized_pos['y'])
            if 'z' in normalized_pos:
                self.osc_client.send_message("/eyetracking/pos/z", normalized_pos['z'])
        except Exception as e:
            print(f"Failed to send OSC data: {e}")

    def check_eyes_open(self, face_landmarks, frame_width, frame_height):
        """
        Check if eyes are open by measuring vertical distance between eyelids
        Returns: True if eyes are open, False if closed
        """
        # Left eye landmarks (upper and lower eyelids)
        left_eye_upper = face_landmarks.landmark[159]  # Upper eyelid
        left_eye_lower = face_landmarks.landmark[145]  # Lower eyelid
        
        # Right eye landmarks (upper and lower eyelids)
        right_eye_upper = face_landmarks.landmark[386]  # Upper eyelid
        right_eye_lower = face_landmarks.landmark[374]  # Lower eyelid
        
        # Calculate normalized vertical distances
        left_eye_opening = abs(left_eye_upper.y - left_eye_lower.y)
        right_eye_opening = abs(right_eye_upper.y - right_eye_lower.y)
        
        # Average eye opening
        avg_eye_opening = (left_eye_opening + right_eye_opening) / 2
        
        # Print the current value to help with threshold adjustment
        print(f"Eye opening value: {avg_eye_opening:.3f}")
        
        # Adjusted threshold based on your values
        self.BLINK_THRESHOLD = 0.005  # Set between your open (0.015) and closed (0.02) values
        
        # Back to original comparison
        return avg_eye_opening > self.BLINK_THRESHOLD
    
    def send_eye_state(self, eyes_open):
        """Send eye state change via OSC"""
        if eyes_open:
            self.osc_client.send_message("/eyetracking/state", "open")
            print("Eyes Open")
        else:
            self.osc_client.send_message("/eyetracking/state", "closed")
            print("Eyes Closed")

    def run(self):
        print("Hold still for calibration...")
        calibration_frames = 30  # Number of frames to average for calibration
        calibration_positions = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Check eye state
                    eyes_open = self.check_eyes_open(face_landmarks, frame_width, frame_height)
                    
                    # If state changed, send message
                    if self.eyes_were_open is None:
                        self.eyes_were_open = eyes_open
                        self.send_eye_state(eyes_open)
                    elif eyes_open != self.eyes_were_open:
                        self.eyes_were_open = eyes_open
                        self.send_eye_state(eyes_open)
                    
                    # Only process eye tracking if eyes are open
                    if eyes_open:
                        left_eye_center = self.get_iris_center(face_landmarks, frame_width, frame_height, left=True)
                        right_eye_center = self.get_iris_center(face_landmarks, frame_width, frame_height, left=False)

                        if left_eye_center and right_eye_center:
                            avg_x = (left_eye_center[0] + right_eye_center[0]) / 2
                            avg_y = (left_eye_center[1] + right_eye_center[1]) / 2
                            
                            # Calculate Z value
                            z_val = self.calculate_z_value(left_eye_center, right_eye_center)
                            
                            # Get normalized position
                            normalized_pos = self.normalize_coordinates(
                                (avg_x, avg_y), 
                                frame_width, 
                                frame_height
                            )
                            if z_val is not None:
                                normalized_pos['z'] = z_val

                            # Handle calibration
                            if not self.calibrated:
                                calibration_positions.append(normalized_pos)
                                if len(calibration_positions) >= calibration_frames:
                                    # Average the calibration positions
                                    avg_pos = {
                                        'x': sum(p['x'] for p in calibration_positions) / len(calibration_positions),
                                        'y': sum(p['y'] for p in calibration_positions) / len(calibration_positions)
                                    }
                                    self.calibrate(avg_pos)
                                    print("Calibration complete! You can now move freely.")
                            else:
                                # Send to Unreal Engine
                                self.send_to_unreal(normalized_pos)

                            # Draw visualization
                            cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 255, 0), -1)
                            cv2.line(frame, 
                                    (int(left_eye_center[0]), int(left_eye_center[1])),
                                    (int(right_eye_center[0]), int(right_eye_center[1])),
                                    (255, 0, 0), 2)
                            
                            # Display values and calibration status
                            status = "CALIBRATING..." if not self.calibrated else ("EYES OPEN" if eyes_open else "EYES CLOSED")
                            cv2.putText(
                                frame,
                                f"{status} X: {normalized_pos['x']:.2f}, Y: {normalized_pos['y']:.2f}, Z: {z_val:.2f}",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                2
                            )

            # Display the frame
            cv2.imshow('Eye Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_iris_center(self, face_landmarks, frame_width, frame_height, left=True):
        """Get iris center coordinates"""
        indices = [468, 469, 470, 471, 472] if left else [473, 474, 475, 476, 477]
        x_coords = []
        y_coords = []
        
        for idx in indices:
            x = int(face_landmarks.landmark[idx].x * frame_width)
            y = int(face_landmarks.landmark[idx].y * frame_height)
            x_coords.append(x)
            y_coords.append(y)

        if x_coords and y_coords:
            x_center = sum(x_coords) // len(x_coords)
            y_center = sum(y_coords) // len(y_coords)
            return (x_center, y_center)
        return None

if __name__ == "__main__":
    tracker = EyeTrackerMediaPipe()
    tracker.run()
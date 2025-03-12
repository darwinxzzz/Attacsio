import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import math
import time

class AttackExercise:
    def __init__(self, hp_per_level=4):
        """
        We fix max_level = 5:
            Level 0 = arms at hips
            Level 3 = arms at shoulders
            Level 5 = arms overhead
        """
        self.max_level = 5
        self.hp_per_level = hp_per_level
        self.previous_level = 0
        self.resetting = False

        # We'll store messages here to animate them (fade, fall, etc.)
        self.active_messages = []
        self.last_message_time = time.time()
        self.message_cooldown = 0.5  # Minimum time between messages in seconds

    def reset(self):
        """Reset the exercise state for a new attack cycle."""
        self.previous_level = 0
        self.resetting = False

    def _compute_level(self, hips_y, shoulders_y, overhead_y, wrists_y):
        """
        Compute a piecewise level [0..5] such that:
          - 0 = wrists at hips
          - 3 = wrists at shoulders
          - 5 = wrists overhead

        We assume hips_y > shoulders_y > overhead_y in typical image coords.
        We'll clamp final level to [0..5].
        """
        # If wrists are below shoulders (wrists_y >= shoulders_y):
        #   ratio_1 = (hips_y - wrists_y) / (hips_y - shoulders_y)
        #   level in [0..3]
        # Else if wrists are above shoulders (wrists_y < shoulders_y):
        #   ratio_2 = (shoulders_y - wrists_y) / (shoulders_y - overhead_y)
        #   level in [3..5]

        # Safety checks in case of unusual angles or detection:
        if hips_y <= shoulders_y or shoulders_y <= overhead_y:
            # Fallback: normal ratio from hips to overhead
            full_range = hips_y - overhead_y
            if full_range <= 0:
                return 0
            arm_lift = hips_y - wrists_y
            ratio = arm_lift / full_range
            ratio = max(0.0, min(ratio, 1.0))
            return int(ratio * 5)

        if wrists_y >= shoulders_y:
            # Segment 1: Hips → Shoulders => Levels 0..3
            segment_1 = hips_y - shoulders_y  # how far from hips to shoulders
            arm_lift_1 = hips_y - wrists_y    # how far from hips to current wrists
            if segment_1 <= 0:
                return 0
            ratio_1 = arm_lift_1 / segment_1
            # Map ratio_1 [0..1] => level [0..3]
            level = ratio_1 * 3
        else:
            # Segment 2: Shoulders → Overhead => Levels 3..5
            segment_2 = shoulders_y - overhead_y
            arm_lift_2 = shoulders_y - wrists_y
            if segment_2 <= 0:
                return 5
            ratio_2 = arm_lift_2 / segment_2
            # Map ratio_2 [0..1] => level [3..5]
            level = 3 + ratio_2 * 2

        # Clamp level to [0..5]
        level = max(0, min(level, 5))
        return int(round(level))

    def process_landmarks(self, landmarks, image_width, image_height):
        """
        Determine the current level of the arms (0..5) using a piecewise approach:
        Hips -> Shoulders -> Overhead.
        """
        try:
            if not landmarks:
                return None

            lm = landmarks.landmark

            # Wrists
            left_wrist = (int(lm[15].x * image_width), int(lm[15].y * image_height))
            right_wrist = (int(lm[16].x * image_width), int(lm[16].y * image_height))
            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2.0

            # Hips
            left_hip = (int(lm[23].x * image_width), int(lm[23].y * image_height))
            right_hip = (int(lm[24].x * image_width), int(lm[24].y * image_height))
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2.0

            # Shoulders
            left_shoulder = (int(lm[11].x * image_width), int(lm[11].y * image_height))
            right_shoulder = (int(lm[12].x * image_width), int(lm[12].y * image_height))
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

            # Overhead (approx top of head)
            # We'll just use nose (landmark 0) or a bit above it.
            # If you prefer, you can use a different landmark or a small offset.
            overhead = (int(lm[0].x * image_width), int(lm[0].y * image_height) - 20)
            overhead_y = overhead[1]

            # For displaying text
            nose = (int(lm[0].x * image_width), int(lm[0].y * image_height))

            # Compute the new level
            current_level = self._compute_level(avg_hip_y, avg_shoulder_y, overhead_y, avg_wrist_y)

            # Throttle message creation to avoid spamming
            current_time = time.time()
            if current_time - self.last_message_time >= self.message_cooldown:
                # Upward movement
                if current_level > self.previous_level:
                    for lvl in range(self.previous_level + 1, current_level + 1):
                        hp_loss = lvl * self.hp_per_level
                        self._add_floating_text(f"Level {lvl} => -{hp_loss} HP",
                                                nose[0], nose[1] - 50)
                        self.last_message_time = current_time

                # Downward movement
                elif current_level < self.previous_level:
                    if current_level == 0 and self.previous_level > 0:
                        # Partial Attack
                        hp_loss = self.previous_level * self.hp_per_level
                        self._add_floating_text(f"Partial Attack! Level {self.previous_level} => -{hp_loss} HP",
                                                nose[0], nose[1] - 50)
                        self.resetting = True
                        self.last_message_time = current_time

                # Full Attack at level 5
                if current_level == 5 and self.previous_level < 5:
                    hp_loss = 5 * self.hp_per_level
                    self._add_floating_text(f"Full Attack => -{hp_loss} HP", nose[0], nose[1] - 50)
                    self.resetting = True
                    self.last_message_time = current_time

            self.previous_level = current_level
            return nose
        except Exception as e:
            print(f"Error processing landmarks: {e}")
            return None

    def finalize_if_needed(self):
        """Reset after a partial or full attack was finalized."""
        if self.resetting:
            self.reset()

    def _add_floating_text(self, text, x, y):
        """
        Add a new floating text message that will appear at (x, y) and animate.
        """
        # Limit the number of active messages to avoid clutter
        if len(self.active_messages) >= 3:
            return

        message_data = {
            "text": text,
            "x": x,
            "y": y,
            "start_time": time.time(),
            "fade_duration": 1.0,   # seconds
            "velocity_y": 1.0       # pixels per frame
        }
        self.active_messages.append(message_data)

    def draw_messages(self, frame):
        """
        Update and draw all active floating messages on the frame.
        - Each message falls slowly and fades out over fade_duration seconds.
        """
        current_time = time.time()
        new_messages = []
        
        for msg in self.active_messages:
            elapsed = current_time - msg["start_time"]
            alpha = 1.0 - (elapsed / msg["fade_duration"])

            if alpha > 0.0:
                # Update position
                msg["y"] += msg["velocity_y"]

                # Fade from red (255) to black (0)
                color_val = int(255 * alpha)
                color = (0, 0, color_val)

                font_scale = 1.2
                thickness = 3

                text_size = cv2.getTextSize(
                    msg["text"], cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, thickness
                )[0]
                text_x = int(msg["x"] - text_size[0] // 2)
                text_y = int(msg["y"] - text_size[1] // 2)

                cv2.putText(
                    frame,
                    msg["text"],
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

                new_messages.append(msg)

        self.active_messages = new_messages


def main():
    # Show a popup before starting camera capture
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Camera Capture", "Starting camera capture. Press 'q' to exit.")
    root.destroy()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )

    # Create the AttackExercise object
    attack_exercise = AttackExercise(hp_per_level=4)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Attack Exercise", cv2.WINDOW_NORMAL)

    prev_frame_time = 0
    target_fps = 30
    frame_interval = 1.0 / target_fps

    while True:
        current_time = time.time()
        elapsed = current_time - prev_frame_time
        if elapsed < frame_interval:
            continue
        prev_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to RGB for pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Convert back to BGR
        draw_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                draw_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Process multi-level attacks
            nose_coords = attack_exercise.process_landmarks(
                results.pose_landmarks, draw_frame.shape[1], draw_frame.shape[0]
            )

        # Draw any floating text messages
        attack_exercise.draw_messages(draw_frame)

        # Overlay "Live Feed" text
        cv2.putText(
            draw_frame,
            "Live Feed",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Attack Exercise", draw_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if we need to reset after partial or full attack
        attack_exercise.finalize_if_needed()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()

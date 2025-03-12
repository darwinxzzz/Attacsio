import math

class AttackExercise:
    def __init__(self, motions_required=5, hp_per_motion=4):
        self.motions_required = motions_required  # Number of motions for a full attack cycle
        self.hp_per_motion = hp_per_motion          # HP deduction per motion
        self.current_motion = 0                     # Count of completed motions
        self.total_hp_deducted = 0                  # Total HP deducted this cycle
        self.state = "waiting"                      # Possible states: waiting, ready, moving
        # Threshold for detecting a "clap" (distance between wrists in pixels)
        self.clap_distance_threshold = 50           # You may need to tune this value
        # For initial detection: arms are considered "down" if wrists are below shoulders.
    
    def reset(self):
        """Reset the exercise state for a new attack cycle."""
        self.current_motion = 0
        self.total_hp_deducted = 0
        self.state = "waiting"
    
    def process_landmarks(self, landmarks, image_width, image_height):
        """
        Process pose landmarks to detect an attack motion.
        Returns a message string if a motion is completed.
        """
        if not landmarks:
            return None

        # MediaPipe landmark indices:
        # Left Shoulder: 11, Right Shoulder: 12, Left Wrist: 15, Right Wrist: 16.
        lm = landmarks.landmark

        left_wrist = (int(lm[15].x * image_width), int(lm[15].y * image_height))
        right_wrist = (int(lm[16].x * image_width), int(lm[16].y * image_height))
        left_shoulder = (int(lm[11].x * image_width), int(lm[11].y * image_height))
        right_shoulder = (int(lm[12].x * image_width), int(lm[12].y * image_height))

        # Use the average shoulder y-coordinate as a reference level.
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

        # Determine if arms are down (ready state) or raised (moving state)
        # For arms down, both wrists should be below shoulders.
        if left_wrist[1] > shoulder_y and right_wrist[1] > shoulder_y:
            self.state = "ready"
        # When both wrists are above shoulders, we consider the arms to be raised.
        elif left_wrist[1] < shoulder_y and right_wrist[1] < shoulder_y:
            if self.state == "ready":
                self.state = "moving"
        
        # In moving state, check if a clap occurs.
        # A clap is detected if the wrists come close enough together.
        if self.state == "moving":
            distance = math.hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1])
            if distance < self.clap_distance_threshold:
                self.current_motion += 1
                self.total_hp_deducted += self.hp_per_motion
                # After a clap, reset state to ready for the next motion.
                self.state = "ready"
                return f"Motion {self.current_motion} complete! -{self.hp_per_motion} HP"
        return None

    def is_cycle_complete(self):
        """Return True if the attack cycle is complete (i.e. 5 motions)."""
        return self.current_motion >= self.motions_required

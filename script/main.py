import cv2
import math
import numpy as np
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import messagebox

# ------------------------------
# 1) Player Class
# ------------------------------
class Player:
    """
    Represents a player in the game with health points and attack capabilities.
    """
    def __init__(self, player_id, max_hp=1000, name="Player"):
        self.id = player_id
        self.name = f"{name} {player_id + 1}"
        self.max_hp = max_hp
        self.current_hp = max_hp
        self.attack_ready = False
        self.attack_power = 0
        self.last_keypoints = None
        self.position = (0, 0)  # For displaying player info
        self.color = (0, 255, 0) if player_id == 0 else (0, 0, 255)  # Green for P1, Red for P2
        
    def take_damage(self, damage):
        """Apply damage to this player and return if they're defeated"""
        self.current_hp = max(0, self.current_hp - damage)
        return self.current_hp <= 0
    
    def reset_attack(self):
        """Reset attack state"""
        self.attack_ready = False
        self.attack_power = 0
    
    def draw_health_bar(self, frame):
        """Draw health bar for this player"""
        if self.position[0] == 0 and self.position[1] == 0:
            return
            
        # Calculate position and dimensions
        x, y = self.position
        bar_width = 100
        bar_height = 10
        
        # Draw name
        cv2.putText(frame, self.name, (x, y - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        
        # Draw health text
        hp_text = f"HP: {self.current_hp}/{self.max_hp}"
        cv2.putText(frame, hp_text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 1)
        
        # Draw background bar (full health)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     (50, 50, 50), -1)
        
        # Draw current health
        health_width = int(bar_width * (self.current_hp / self.max_hp))
        cv2.rectangle(frame, (x, y), (x + health_width, y + bar_height), 
                     self.color, -1)
        
        # If attack is ready, show indicator
        if self.attack_ready:
            attack_text = f"ATTACK READY: {self.attack_power} DMG"
            cv2.putText(frame, attack_text, (x, y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

# ------------------------------
# 2) AttackExercise Class
# ------------------------------
class AttackExercise:
    """
    Buddha Clap exercise with physiotherapy elements for elderly users:
    
    Piecewise arm raising from hips (level=0) to overhead (level=5).
    - Segment 1: 0..3 (hips -> shoulders)
    - Segment 2: 3..5 (shoulders -> overhead)
    
    Added physiotherapy elements:
    - Validates shoulder range of motion
    - Ensures good posture and alignment
    - Monitors for excessive strain or compensation
    - Provides elderly-appropriate feedback and encouragement
    """

    def __init__(self, max_level=5, hp_per_level=4):
        """
        Args:
            max_level (int): The maximum level (5 means fully overhead).
            hp_per_level (int): HP deducted per level (4 => level 5 = 20 HP).
        """
        self.max_level = max_level
        self.hp_per_level = hp_per_level
        self.players = []
        
        # For floating text
        self.active_messages = []
        self.message_cooldown = 1.0  # Longer for elderly users
        self.last_message_time = time.time()
        
        # Physiotherapy validation parameters
        self.min_hold_time = 1.5  # seconds to hold at a level
        self.posture_threshold = 30  # pixels for alignment checks
        self.shoulder_pain_zone = 160  # degrees (avoid full overhead for elderly)
        
    def add_player(self, player):
        """Add a player to track"""
        self.players.append({
            "player": player, 
            "current_level": 0, 
            "resetting": False,
            "hold_start_time": 0,
            "is_holding": False,
            "posture_feedback": "",
            "last_posture_check": time.time()
        })

    def reset_player(self, player_id):
        """Reset the state for a player's new attack cycle."""
        if player_id < len(self.players):
            self.players[player_id]["current_level"] = 0
            self.players[player_id]["resetting"] = False
            self.players[player_id]["hold_start_time"] = 0
            self.players[player_id]["is_holding"] = False
            self.players[player_id]["posture_feedback"] = ""
            self.players[player_id]["player"].reset_attack()

    def _compute_level(self, hips_y, shoulders_y, overhead_y, wrists_y):
        """
        Piecewise:
          - 0..3: hips -> shoulders (more gradual for elderly)
          - 3..5: shoulders -> overhead (limited for elderly safety)
        Shoulders forced to be midpoint level=3.
        """
        try:
            # If geometry is strange, fallback to single ratio
            if hips_y <= shoulders_y or shoulders_y <= overhead_y:
                full_range = hips_y - overhead_y
                if full_range <= 0:
                    return 0
                arm_lift = hips_y - wrists_y
                ratio = max(0.0, min(arm_lift / full_range, 1.0))
                return int(round(ratio * 5))

            # If wrists below shoulders => Segment 1 => level in [0..3]
            if wrists_y >= shoulders_y:
                segment_1 = hips_y - shoulders_y
                arm_lift_1 = hips_y - wrists_y
                if segment_1 <= 0:
                    return 0
                ratio_1 = max(0.0, min(arm_lift_1 / segment_1, 1.0))
                level = ratio_1 * 3
            else:
                # wrists above shoulders => Segment 2 => level in [3..5]
                segment_2 = shoulders_y - overhead_y
                arm_lift_2 = shoulders_y - wrists_y
                if segment_2 <= 0:
                    return 5
                ratio_2 = max(0.0, min(arm_lift_2 / segment_2, 1.0))
                level = 3 + ratio_2 * 2

            return int(round(max(0, min(level, 5))))
        except Exception as e:
            print(f"Error in _compute_level: {e}")
            return 0

    def check_posture(self, keypoints):
        """Check for proper posture during arm raising"""
        try:
            # Check if shoulders are level
            left_shoulder_y = keypoints[5][1]
            right_shoulder_y = keypoints[6][1]
            shoulder_deviation = abs(left_shoulder_y - right_shoulder_y)
            
            # Check head alignment with shoulders
            nose_x = keypoints[0][0]
            left_shoulder_x = keypoints[5][0]
            right_shoulder_x = keypoints[6][0]
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
            head_deviation = abs(nose_x - shoulder_center_x)
            
            # Generate feedback
            feedback = []
            if shoulder_deviation > self.posture_threshold:
                feedback.append("Keep shoulders level")
            
            if head_deviation > self.posture_threshold:
                feedback.append("Align head with shoulders")
            
            return len(feedback) == 0, feedback
        except Exception as e:
            print(f"Error checking posture: {e}")
            return True, []  # Default to allowing movement
    
    def check_shoulder_strain(self, keypoints, wrists_y, shoulders_y):
        """Check for potential shoulder strain in elderly users"""
        try:
            # For elderly, limit overhead extension to prevent shoulder impingement
            if wrists_y < shoulders_y - 100:  # Wrists significantly above shoulders
                # Calculate arm angle
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]
                
                # Very simple estimation of arm angle from vertical
                left_angle = abs(math.degrees(math.atan2(
                    left_wrist[0] - left_shoulder[0], 
                    left_shoulder[1] - left_wrist[1])))
                right_angle = abs(math.degrees(math.atan2(
                    right_wrist[0] - right_shoulder[0], 
                    right_shoulder[1] - right_wrist[1])))
                
                arm_angle = max(left_angle, right_angle)
                
                # Check if arms are in a potential impingement zone
                if arm_angle > self.shoulder_pain_zone:
                    return False, "Avoid full overhead - protect shoulders"
            
            return True, ""
        except Exception as e:
            print(f"Error checking shoulder strain: {e}")
            return True, ""  # Default to allowing movement

    def process_keypoints(self, player_id, keypoints):
        """
        Process keypoints for a specific player with physiotherapy validation.
        Returns attack message and target player ID if an attack is triggered.
        """
        try:
            if player_id >= len(self.players) or keypoints is None or keypoints.shape[0] < 17:
                return None, None
            
            player_data = self.players[player_id]
            player = player_data["player"]
            
            # Save last keypoints
            player.last_keypoints = keypoints
            
            # Find the opponent player_id
            target_id = 1 if player_id == 0 else 0
            
            # Indices for YOLO keypoints
            LEFT_HIP, RIGHT_HIP = 11, 12
            LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
            LEFT_WRIST, RIGHT_WRIST = 9, 10
            NOSE = 0

            # Extract points, skip if near (0,0)
            if any(x < 2 and y < 2 for (x, y) in keypoints[[LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]]):
                return None, None

            lhx, lhy = keypoints[LEFT_HIP]
            rhx, rhy = keypoints[RIGHT_HIP]
            lsx, lsy = keypoints[LEFT_SHOULDER]
            rsx, rsy = keypoints[RIGHT_SHOULDER]
            lwx, lwy = keypoints[LEFT_WRIST]
            rwx, rwy = keypoints[RIGHT_WRIST]
            nx, ny = keypoints[NOSE]
            
            # Update player position for UI
            player.position = (int(nx) - 50, int(ny) - 100)

            # Averages
            avg_hip_y = (lhy + rhy) / 2
            avg_shoulder_y = (lsy + rsy) / 2
            avg_wrist_y = (lwy + rwy) / 2
            overhead_y = ny - 20

            # Check posture - only check periodically to avoid too many corrections
            current_time = time.time()
            posture_check_interval = 2.0  # seconds
            if current_time - player_data["last_posture_check"] > posture_check_interval:
                good_posture, posture_feedback = self.check_posture(keypoints)
                player_data["last_posture_check"] = current_time
                
                if not good_posture:
                    player_data["posture_feedback"] = " & ".join(posture_feedback)
                    if current_time - self.last_message_time >= self.message_cooldown:
                        self._add_floating_text(
                            f"Posture tip: {player_data['posture_feedback']}", 
                            nx, ny - 50, 
                            (0, 140, 255)  # Softer orange for elderly
                        )
                        self.last_message_time = current_time
                else:
                    player_data["posture_feedback"] = ""
            
            # Check for shoulder strain
            safe_shoulders, strain_feedback = self.check_shoulder_strain(
                keypoints, avg_wrist_y, avg_shoulder_y
            )
            
            if not safe_shoulders and current_time - self.last_message_time >= self.message_cooldown:
                self._add_floating_text(
                    f"Safety tip: {strain_feedback}", 
                    nx, ny - 80, 
                    (0, 0, 255)  # Red for important safety
                )
                self.last_message_time = current_time
                # Skip further processing if unsafe movement detected
                return None, None

            new_level = self._compute_level(avg_hip_y, avg_shoulder_y, overhead_y, avg_wrist_y)
            current_level = player_data["current_level"]

            # Handle level transitions with hold time validation
            attack_message = None
            target_player_id = None
            
            if current_time - self.last_message_time >= self.message_cooldown:
                # If player moved up in level
                if new_level > current_level:
                    # Start hold timer for new level
                    player_data["hold_start_time"] = current_time
                    player_data["is_holding"] = True
                    
                    for lvl in range(current_level + 1, new_level + 1):
                        hp_loss = lvl * self.hp_per_level
                        self._add_floating_text(
                            f"{player.name}: Level {lvl} - Hold position", 
                            nx, ny - 50, 
                            player.color
                        )
                        self.last_message_time = current_time

                # If player reached max level => full attack (only after hold time)
                elif new_level == self.max_level and current_level == self.max_level:
                    hold_time = current_time - player_data["hold_start_time"]
                    
                    if hold_time >= self.min_hold_time and player_data["is_holding"]:
                        player.attack_power = self.max_level * self.hp_per_level
                        player.attack_ready = True
                        attack_message = f"{player.name} FULL ATTACK: {player.attack_power} DMG"
                        target_player_id = target_id
                        self._add_floating_text(
                            f"Well done! Full movement completed", 
                            nx, ny - 50, 
                            (0, 255, 0)  # Green for success
                        )
                        player_data["resetting"] = True
                        player_data["is_holding"] = False
                        self.last_message_time = current_time
                
                # If player dropped from n>0 back to 0 => partial attack
                elif new_level < current_level:
                    if new_level == 0 and current_level > 0:
                        hold_time = current_time - player_data["hold_start_time"]
                        
                        if hold_time >= self.min_hold_time and player_data["is_holding"]:
                            player.attack_power = current_level * self.hp_per_level
                            player.attack_ready = True
                            attack_message = f"{player.name} ATTACK: {player.attack_power} DMG"
                            target_player_id = target_id
                            self._add_floating_text(
                                f"Good job! Movement completed", 
                                nx, ny - 50, 
                                (0, 255, 0)  # Green for success
                            )
                            player_data["resetting"] = True
                            player_data["is_holding"] = False
                            self.last_message_time = current_time
                        else:
                            # If didn't hold long enough, provide feedback
                            self._add_floating_text(
                                "Hold position longer next time", 
                                nx, ny - 50, 
                                (255, 255, 0)  # Yellow for advice
                            )
                            self.last_message_time = current_time
                            player_data["is_holding"] = False

            player_data["current_level"] = new_level
            return attack_message, target_player_id
            
        except Exception as e:
            print(f"Error processing keypoints for player {player_id}: {e}")
            return None, None

    def finalize_if_needed(self, player_id):
        """Reset after a partial or full attack was finalized."""
        if player_id < len(self.players) and self.players[player_id]["resetting"]:
            self.reset_player(player_id)
            return True
        return False

    def _add_floating_text(self, text, x, y, color=(0, 0, 255)):
        """Add a floating text message to the screen"""
        try:
            if len(self.active_messages) >= 5:
                return
                
            msg_data = {
                "text": text,
                "x": x,
                "y": y,
                "start_time": time.time(),
                "fade_duration": 2.0,  # Longer duration for elderly to read
                "velocity_y": 0.7,     # Slower movement for readability
                "color": color
            }
            self.active_messages.append(msg_data)
        except Exception as e:
            print(f"Error adding floating text: {e}")

    def draw_messages(self, frame):
        """Draw all active floating text messages with larger font for elderly users"""
        try:
            current_time = time.time()
            new_msgs = []
            for msg in self.active_messages:
                elapsed = current_time - msg["start_time"]
                alpha = 1.0 - (elapsed / msg["fade_duration"])
                if alpha > 0.0:
                    msg["y"] += msg["velocity_y"]
                    
                    # Get base color from message
                    base_color = msg.get("color", (0, 0, 255))
                    
                    # Apply alpha to the color
                    color = tuple(int(c * alpha) for c in base_color)

                    # Larger font for elderly users
                    font_scale = 0.8  
                    thickness = 2

                    text_size = cv2.getTextSize(msg["text"], cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale, thickness)[0]
                    text_x = int(msg["x"] - text_size[0] // 2)
                    text_y = int(msg["y"] - text_size[1] // 2)

                    # Draw with black outline for better visibility
                    cv2.putText(frame,
                                msg["text"],
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (0, 0, 0),  # Black outline
                                thickness + 2,
                                cv2.LINE_AA)
                    
                    # Draw main text
                    cv2.putText(frame,
                                msg["text"],
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                color,
                                thickness,
                                cv2.LINE_AA)
                    new_msgs.append(msg)
            self.active_messages = new_msgs
        except Exception as e:
            print(f"Error drawing messages: {e}")
            self.active_messages = []


# ------------------------------
# 3) YoloPoseEstimator Class
# ------------------------------
class YoloPoseEstimator:
    """
    Uses YOLOv8 Pose for real-time pose estimation,
    draws skeleton lines, big joint circles.
    """

    SKELETON = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (11, 12),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    def __init__(self, model_path='yolov8n-pose.pt', conf=0.5):
        try:
            self.model = YOLO(model_path)
            self.conf = conf
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            raise

    def estimate_pose(self, frame):
        """Estimate pose for all people in the frame"""
        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            if not results or len(results) == 0:
                return []
                
            result = results[0]
            if not hasattr(result, "keypoints") or result.keypoints is None:
                return []
                
            kp_data = result.keypoints.xy.cpu().numpy()  # shape [N,17,2]
            return kp_data
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return []

    def draw_skeleton(self, frame, keypoints_list, player_colors=None):
        """Draw skeletons for each person with player-specific colors"""
        try:
            for i, person in enumerate(keypoints_list):
                if person.shape[0] < 17:
                    continue
                    
                # Use player color if available, otherwise default colors
                color = player_colors[i] if player_colors and i < len(player_colors) else (0, 255, 255)
                
                # Draw skeleton lines
                for (p1, p2) in self.SKELETON:
                    x1, y1 = person[p1]
                    x2, y2 = person[p2]
                    if (x1 < 2 and y1 < 2) or (x2 < 2 and y2 < 2):
                        continue
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            color, 2)
                
                # Draw joints
                for (x, y) in person:
                    if x < 2 and y < 2:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        except Exception as e:
            print(f"Error drawing skeleton: {e}")


# ------------------------------
# 4) GameManager Class
# ------------------------------
class GameManager:
    """
    Manages the overall game state, players, exercise transitions, and UI.
    
    Enhanced for elderly users with:
    - Clear, larger text and UI elements
    - Simplified controls and instructions
    - Gentler timing and progression
    - Comprehensive exercise program with physiotherapy elements
    """
    def __init__(self, num_players=2):
        # Initialize players
        self.players = [Player(i, max_hp=1000) for i in range(num_players)]
        
        # Initialize exercise classes
        self.attack_exercise = AttackExercise(max_level=5, hp_per_level=4)
        self.side_stretch_left = SideStretch()
        self.side_stretch_right = SideStretch()
        self.squat_exercise = SquatExercise()
        
        # Exercise selection
        self.current_exercise = 'attack'  # Set default exercise to 'attack'
        self.exercise_names = {
            'attack': 'Arm Raising',
            'side_stretch': 'Side Stretch',
            'squat': 'Chair Squat'
        }
        
        # Game state
        self.game_over = False
        self.winner = None
        self.session_start_time = time.time()
        self.exercise_time = {
            'attack': 0,
            'side_stretch': 0,
            'squat': 0
        }
        self.last_exercise_switch_time = time.time()
        
        # Elderly-friendly UI settings
        self.large_font_scale = 0.9
        self.font_thickness = 2
        self.high_contrast_colors = {
            'title': (255, 255, 255),  # White
            'instruction': (220, 220, 220),  # Light gray
            'positive': (50, 200, 50),  # Softer green
            'warning': (50, 150, 230)   # Softer orange (not harsh red)
        }
        
        # Initialize exercises for both sides
        self.side_stretch_left.set_stretch_side('left')
        self.side_stretch_right.set_stretch_side('right')
        
        # Add players to attack exercise
        for player in self.players:
            self.attack_exercise.add_player(player)
    
    def process_frame(self, frame, keypoints_list):
        """Process a frame and update game state based on the current exercise."""
        try:
            # Update exercise timing
            current_time = time.time()
            elapsed = current_time - self.last_exercise_switch_time
            self.exercise_time[self.current_exercise] += elapsed
            self.last_exercise_switch_time = current_time
            
            # Process each player's movement based on the current exercise
            for i, player in enumerate(self.players):
                if i < len(keypoints_list):
                    if self.current_exercise == 'attack':
                        attack_msg, target_id = self.attack_exercise.process_keypoints(i, keypoints_list[i])
                        
                        # Apply attack if one was triggered
                        if attack_msg and target_id is not None and target_id < len(self.players):
                            target_player = self.players[target_id]
                            attack_power = player.attack_power
                            
                            # Apply damage to target player
                            is_defeated = target_player.take_damage(attack_power)
                            
                            # Check for game over
                            if is_defeated:
                                self.game_over = True
                                self.winner = player
                                
                        # Reset attack state if needed        
                        self.attack_exercise.finalize_if_needed(i)
                        
                    elif self.current_exercise == 'side_stretch':
                        # Process side stretch for the appropriate player and side
                        if i == 0:  # Player 1 does left side stretch
                            self.side_stretch_left.update_stretch(keypoints_list[i])
                        elif i == 1:  # Player 2 does right side stretch
                            self.side_stretch_right.update_stretch(keypoints_list[i])
                            
                    elif self.current_exercise == 'squat':
                        # Process squat exercise for each player
                        self.squat_exercise.update_squat(keypoints_list[i])

            # Draw health bars and game state
            self._draw_game_ui(frame)
            
        except Exception as e:
            print(f"Error in game processing: {e}")
    
    def _draw_game_ui(self, frame):
        """Draw elderly-friendly game UI elements"""
        try:
            # Draw health bars for each player
            for player in self.players:
                player.draw_health_bar(frame)
            
            # Draw session information in top-left corner
            session_time = time.time() - self.session_start_time
            minutes, seconds = divmod(int(session_time), 60)
            session_text = f"Exercise Time: {minutes:02d}:{seconds:02d}"
            
            cv2.putText(frame, session_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       self.large_font_scale, self.high_contrast_colors['title'], self.font_thickness)
            
            # Draw current exercise with larger font
            if self.current_exercise:
                exercise_name = self.exercise_names.get(self.current_exercise, self.current_exercise.capitalize())
                exercise_text = f"Current Exercise: {exercise_name}"
                cv2.putText(frame, exercise_text, 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           self.large_font_scale, self.high_contrast_colors['title'], self.font_thickness)
            
            # Draw keyboard shortcuts - larger and clearer for elderly
            shortcuts_y = frame.shape[0] - 60
            cv2.putText(frame, "Controls: 1=Arm Raise | 2=Side Stretch | 3=Squat | r=Reset | q=Quit", 
                       (10, shortcuts_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, self.high_contrast_colors['instruction'], 1)

            # Draw exercise-specific UI elements
            if self.current_exercise == 'side_stretch':
                # Display stretch information and feedback for both players
                if len(self.players) >= 1:
                    # Left side stretch feedback (Player 1)
                    self.side_stretch_left.draw_feedback(frame, position=(20, 120))
                    left_score = self.side_stretch_left.score
                    cv2.putText(frame, f"Player 1 Score: {left_score}", 
                                (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, self.players[0].color, self.font_thickness)
                
                if len(self.players) >= 2:
                    # Right side stretch feedback (Player 2)
                    self.side_stretch_right.draw_feedback(frame, position=(frame.shape[1] - 300, 120))
                    right_score = self.side_stretch_right.score
                    cv2.putText(frame, f"Player 2 Score: {right_score}", 
                                (frame.shape[1] - 300, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, self.players[1].color, self.font_thickness)
            
            elif self.current_exercise == 'squat':
                # Show squat exercise feedback
                self.squat_exercise.draw_feedback(frame, position=(30, 120))
                
                # Draw score
                squat_score = self.squat_exercise.score
                cv2.putText(frame, f"Squat Score: {squat_score}", 
                           (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, self.high_contrast_colors['positive'], self.font_thickness)

            # Draw game state if game is over
            if self.game_over and self.winner:
                # Draw semi-transparent overlay for better readability
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0]//2 - 60), 
                             (frame.shape[1], frame.shape[0]//2 + 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                winner_text = f"{self.winner.name} WINS!"
                cv2.putText(frame, winner_text, 
                            (frame.shape[1]//2 - 150, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.winner.color, 3)
                
                # Instruction to restart - larger for elderly users
                cv2.putText(frame, "Press 'r' to restart", 
                            (frame.shape[1]//2 - 150, frame.shape[0]//2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.high_contrast_colors['instruction'], 2)
                            
            # Add helpful elderly-focused reminder about taking breaks
            if session_time > 300 and session_time % 300 < 10:  # Every 5 minutes
                reminder_text = "Remember to take a short break if needed"
                cv2.putText(frame, reminder_text, 
                           (frame.shape[1]//2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, self.high_contrast_colors['warning'], self.font_thickness)
                
        except Exception as e:
            print(f"Error drawing game UI: {e}")
    
    def restart_game(self):
        """Reset the game state for a new round"""
        for i, player in enumerate(self.players):
            player.current_hp = player.max_hp
            player.reset_attack()
            self.attack_exercise.reset_player(i)
        
        # Reset side stretch scores
        self.side_stretch_left.score = 0
        self.side_stretch_right.score = 0
        self.side_stretch_left.completed_stretches = {1: 0, 2: 0, 3: 0}
        self.side_stretch_right.completed_stretches = {1: 0, 2: 0, 3: 0}
        
        # Reset squat exercise
        self.squat_exercise.score = 0
        self.squat_exercise.completed_squats = {1: 0, 2: 0, 3: 0}
            
        self.game_over = False
        self.winner = None
    
    def switch_exercise(self, exercise):
        """Switch the current exercise."""
        old_exercise = self.current_exercise
        self.current_exercise = exercise
        self.last_exercise_switch_time = time.time()
        print(f"Switched from {old_exercise} to {exercise} exercise")


# ------------------------------
# 5) SideStretch Exercise Class
# ------------------------------
class SideStretch:
    """
    Side stretch exercise for physiotherapy:
    Level 1: 15° lateral bend - Beginners
    Level 2: 30° lateral bend - Intermediate
    Level 3: 45° lateral bend - Advanced
    
    Must maintain proper posture (straight spine) and hold stretch for minimum time.
    """
    def __init__(self):
        # Angles for different levels (in degrees)
        self.levels = {1: 15, 2: 30, 3: 45}  
        self.current_level = 0
        self.score = 0
        self.stretch_side = None  # 'left' or 'right'
        
        # Stretch validation parameters
        self.hold_time_required = 2.0  # seconds to hold at a level
        self.stretch_start_time = 0
        self.is_stretching = False
        self.last_stretch_level = 0
        
        # Feedback messages
        self.feedback = ""
        self.feedback_color = (255, 255, 255)
        self.last_message_time = time.time()
        self.message_cooldown = 1.0
        
        # For tracking stretch history
        self.completed_stretches = {1: 0, 2: 0, 3: 0}  # level: count
        
    def calculate_side_angle(self, shoulder, hip, keypoints):
        """
        Calculate the angle of lateral bend.
        For left stretch: right shoulder to right hip to vertical
        For right stretch: left shoulder to left hip to vertical
        """
        try:
            # Get points
            if self.stretch_side == 'left':
                # For left stretch, we measure from right side of body
                s_point = keypoints[6]  # right shoulder
                h_point = keypoints[12]  # right hip
            else:
                # For right stretch, we measure from left side of body
                s_point = keypoints[5]  # left shoulder
                h_point = keypoints[11]  # left hip
                
            # Create vertical reference line
            vertical_point = (h_point[0], h_point[1] - 100)  # 100px up from hip
            
            # Convert to numpy arrays
            s = np.array(s_point)
            h = np.array(h_point)
            v = np.array(vertical_point)
            
            # Calculate vectors
            sh_vec = s - h  # shoulder to hip vector
            vh_vec = v - h  # vertical to hip vector
            
            # Calculate angle
            dot_product = np.dot(sh_vec, vh_vec)
            norm_sh = np.linalg.norm(sh_vec)
            norm_vh = np.linalg.norm(vh_vec)
            
            cos_angle = dot_product / (norm_sh * norm_vh)
            # Clamp cos_angle to [-1, 1] to avoid numerical errors
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating side angle: {e}")
            return 0
            
    def check_posture(self, keypoints):
        """
        Validate proper posture for side stretch.
        - Shoulders should be aligned
        - Hips should be stable
        - Spine should be straight except for the lateral bend
        """
        try:
            # For simplicity, check if shoulders are level (y-coord)
            left_shoulder_y = keypoints[5][1]
            right_shoulder_y = keypoints[6][1]
            
            # Check if hips are level (y-coord)
            left_hip_y = keypoints[11][1]
            right_hip_y = keypoints[12][1]
            
            # Calculate deviations
            shoulder_deviation = abs(left_shoulder_y - right_shoulder_y)
            hip_deviation = abs(left_hip_y - right_hip_y)
            
            # Define thresholds (in pixels)
            shoulder_threshold = 30
            hip_threshold = 20
            
            # Check if posture is good
            good_posture = True
            posture_feedback = []
            
            if shoulder_deviation > shoulder_threshold:
                good_posture = False
                posture_feedback.append("Keep shoulders level")
                
            if hip_deviation > hip_threshold:
                good_posture = False
                posture_feedback.append("Keep hips stable")
            
            return good_posture, posture_feedback
            
        except Exception as e:
            print(f"Error checking posture: {e}")
            return False, ["Could not validate posture"]
            
    def update_stretch(self, keypoints):
        """Update the stretch level based on keypoints and validate movement."""
        try:
            if keypoints is None or keypoints.shape[0] < 17:
                return
                
            # Calculate bend angle
            angle = self.calculate_side_angle(None, None, keypoints)
            
            # Check posture
            good_posture, posture_feedback = self.check_posture(keypoints)
            
            # Determine level based on angle
            current_level = 0
            for level, threshold in self.levels.items():
                if angle >= threshold:
                    current_level = level
            
            # Update current level
            self.current_level = current_level
            
            current_time = time.time()
            
            # If not in correct posture, reset timer
            if not good_posture:
                if current_time - self.last_message_time >= self.message_cooldown:
                    self.feedback = " & ".join(posture_feedback)
                    self.feedback_color = (0, 0, 255)  # Red for correction
                    self.last_message_time = current_time
                self.is_stretching = False
                self.stretch_start_time = 0
                return
                
            # Handle level transitions
            if current_level > 0:
                # Started or continuing a stretch
                if not self.is_stretching or self.last_stretch_level != current_level:
                    # New stretch or changed level
                    self.is_stretching = True
                    self.stretch_start_time = current_time
                    self.last_stretch_level = current_level
                    self.feedback = f"Holding Level {current_level} stretch..."
                    self.feedback_color = (255, 255, 0)  # Yellow for in-progress
                elif self.is_stretching:
                    # Continuing same level stretch
                    hold_duration = current_time - self.stretch_start_time
                    
                    # Check if held long enough
                    if hold_duration >= self.hold_time_required:
                        # Complete the stretch if we haven't given points for this hold yet
                        if self.last_stretch_level == current_level:
                            # Award points
                            points = 10 * current_level
                            self.score += points
                            self.completed_stretches[current_level] += 1
                            
                            # Update feedback
                            self.feedback = f"Level {current_level} complete! +{points} points"
                            self.feedback_color = (0, 255, 0)  # Green for success
                            self.last_message_time = current_time
                            
                            # Reset for next stretch
                            self.is_stretching = False
                            self.last_stretch_level = 0
            else:
                # Not actively stretching
                self.is_stretching = False
                
        except Exception as e:
            print(f"Error updating stretch: {e}")
            
    def draw_feedback(self, frame, position=(20, 120)):
        """Draw stretch feedback on the frame."""
        try:
            if not self.feedback:
                return
                
            # Draw the feedback text
            cv2.putText(frame, self.feedback, position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.feedback_color, 2)
                       
            # Draw stretch side indicator
            side_text = f"Stretch Side: {self.stretch_side.capitalize()}" if self.stretch_side else ""
            if side_text:
                y_offset = position[1] + 30
                cv2.putText(frame, side_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # If actively stretching, show hold timer
            if self.is_stretching:
                hold_time = time.time() - self.stretch_start_time
                timer_text = f"Hold: {hold_time:.1f}s / {self.hold_time_required:.1f}s"
                y_offset = position[1] + 60
                
                # Color changes as time progresses
                progress = min(1.0, hold_time / self.hold_time_required)
                timer_color = (
                    int(255 * (1 - progress)),  # R decreases
                    int(255 * progress),       # G increases
                    0                          # B stays 0
                )
                
                cv2.putText(frame, timer_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
                           
            # Show statistics
            stats_text = f"Completed: L1:{self.completed_stretches[1]} L2:{self.completed_stretches[2]} L3:{self.completed_stretches[3]}"
            y_offset = position[1] + 90
            cv2.putText(frame, stats_text, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
        except Exception as e:
            print(f"Error drawing stretch feedback: {e}")

    def set_stretch_side(self, side):
        """Set the side of the stretch to monitor."""
        self.stretch_side = side
        self.feedback = f"Ready for {side} side stretch"
        self.feedback_color = (255, 255, 255)


# ------------------------------
# 6) SquatExercise Class
# ------------------------------
class SquatExercise:
    """
    Squat exercise designed specifically for elderly users:
    
    Level 1: 15° knee bend - Very gentle mobility exercise
    Level 2: 30° knee bend - Mild strengthening
    Level 3: 45° knee bend - Moderate strengthening
    
    Safety features:
    - Validates proper alignment of knees (not extending beyond toes)
    - Monitors balance and stability
    - Ensures proper posture and back alignment
    - Slower hold times appropriate for elderly users
    - Clear feedback and encouragement
    """
    def __init__(self):
        # Angles for different squat depths (in degrees)
        self.levels = {
            1: {'knee_angle': 165, 'name': 'Gentle'},  # 15° bend (180-15)
            2: {'knee_angle': 150, 'name': 'Moderate'},  # 30° bend
            3: {'knee_angle': 135, 'name': 'Deep'}    # 45° bend
        }
        
        self.current_level = 0
        self.score = 0
        
        # Validation parameters appropriate for elderly users
        self.hold_time_required = 3.0  # longer hold time for stability
        self.rep_timeout = 10.0  # longer timeout between reps
        self.squat_start_time = 0
        self.is_squatting = False
        self.last_squat_level = 0
        self.last_rep_time = 0
        
        # Proper form thresholds
        self.knee_alignment_threshold = 40  # pixels for knee-toe alignment
        self.balance_threshold = 30  # pixels for side-to-side movement
        
        # Feedback messages
        self.feedback = "Prepare for squat exercise"
        self.detailed_feedback = ""
        self.feedback_color = (255, 255, 255)
        self.last_message_time = time.time()
        self.message_cooldown = 2.0  # longer cooldown for elderly users
        
        # For tracking history
        self.completed_squats = {1: 0, 2: 0, 3: 0}  # level: count
        self.total_time_in_squat = 0
        self.max_knee_angle = 180  # track lowest squat
    
    def calculate_knee_angle(self, keypoints):
        """
        Calculate the angle formed at the knee joint.
        Angle between hip, knee, and ankle.
        """
        try:
            # Use average of both legs
            left_hip = keypoints[11]  # left hip
            right_hip = keypoints[12]  # right hip
            left_knee = keypoints[13]  # left knee
            right_knee = keypoints[14]  # right knee
            left_ankle = keypoints[15]  # left ankle
            right_ankle = keypoints[16]  # right ankle
            
            # Calculate angle for each leg
            left_angle = self._calculate_joint_angle(left_hip, left_knee, left_ankle)
            right_angle = self._calculate_joint_angle(right_hip, right_knee, right_ankle)
            
            # Use average of both legs
            avg_angle = (left_angle + right_angle) / 2
            
            return avg_angle
            
        except Exception as e:
            print(f"Error calculating knee angle: {e}")
            return 180  # default to straight legs
    
    def _calculate_joint_angle(self, p1, p2, p3):
        """Calculate angle formed by three points with p2 as the vertex"""
        try:
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Protect against floating point errors
            
            angle = np.degrees(np.arccos(cosine_angle))
            return angle
        except:
            return 180  # Default to straight legs on error
    
    def check_knee_alignment(self, keypoints):
        """
        Check if knees are properly aligned (not extending beyond toes).
        Critical for elderly users to prevent knee strain.
        """
        try:
            # Get knee and ankle positions
            left_knee_x = keypoints[13][0]
            right_knee_x = keypoints[14][0]
            left_ankle_x = keypoints[15][0]
            right_ankle_x = keypoints[16][0]
            
            # Calculate how far knees extend beyond ankles (negative = behind, positive = beyond)
            left_extension = left_knee_x - left_ankle_x
            right_extension = right_knee_x - right_ankle_x
            
            # Check if knees extend too far beyond toes (adjusted for camera perspective)
            # For elderly users, we want to be very conservative
            knees_aligned = (left_extension < self.knee_alignment_threshold and 
                            right_extension < self.knee_alignment_threshold)
            
            if not knees_aligned:
                return False, "Keep knees behind toes"
            
            return True, ""
            
        except Exception as e:
            print(f"Error checking knee alignment: {e}")
            return False, "Cannot validate knee position"
    
    def check_balance(self, keypoints):
        """
        Check if user is maintaining proper balance during squat.
        Elderly users need to maintain good stability.
        """
        try:
            # Use shoulder and hip position to detect leaning
            left_shoulder_x = keypoints[5][0]
            right_shoulder_x = keypoints[6][0]
            left_hip_x = keypoints[11][0]
            right_hip_x = keypoints[12][0]
            
            # Calculate center points
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
            hip_center_x = (left_hip_x + right_hip_x) / 2
            
            # Check if shoulders are aligned over hips (detect leaning)
            offset = abs(shoulder_center_x - hip_center_x)
            
            if offset > self.balance_threshold:
                return False, "Maintain balance - keep back straight"
            
            return True, ""
            
        except Exception as e:
            print(f"Error checking balance: {e}")
            return False, "Cannot validate balance"
    
    def check_posture(self, keypoints):
        """
        Check overall posture during squat.
        Validates back alignment and ensures user isn't hunching forward.
        """
        try:
            # Use alignment of shoulders, hips, and knees to check back positioning
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            # Calculate back angle (should be relatively vertical)
            shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                             (left_shoulder[1] + right_shoulder[1])/2]
            hip_center = [(left_hip[0] + right_hip[0])/2, 
                        (left_hip[1] + right_hip[1])/2]
            
            # Vector pointing up from hips to shoulders
            back_vector = [shoulder_center[0] - hip_center[0], 
                          shoulder_center[1] - hip_center[1]]
            
            # Ideal upright vector
            upright_vector = [0, -1]  # Pointing up
            
            # Calculate angle
            dot_product = back_vector[0] * upright_vector[0] + back_vector[1] * upright_vector[1]
            magnitude = math.sqrt(back_vector[0]**2 + back_vector[1]**2)
            
            # Angle between back and vertical
            if magnitude > 0:
                cosine = dot_product / magnitude
                cosine = max(-1.0, min(1.0, cosine))
                angle = math.degrees(math.acos(cosine))
                
                # For elderly users, we want to be conservative
                if angle > 30:  # More than 30 degrees from vertical
                    return False, "Keep back straight"
            
            return True, ""
            
        except Exception as e:
            print(f"Error checking posture: {e}")
            return False, "Cannot validate posture"
    
    def update_squat(self, keypoints):
        """Update squat exercise state based on keypoints and validate form."""
        try:
            if keypoints is None or keypoints.shape[0] < 17:
                return

            # Calculate knee angle (180 = straight, smaller = more bend)
            knee_angle = self.calculate_knee_angle(keypoints)
            
            # Track max bend
            self.max_knee_angle = min(self.max_knee_angle, knee_angle)
            
            # Determine squat level based on knee angle
            current_level = 0
            for level, data in self.levels.items():
                if knee_angle <= data['knee_angle']:
                    current_level = level
            
            # Update current level
            self.current_level = current_level
            
            # Check form
            form_issues = []
            knee_aligned, knee_feedback = self.check_knee_alignment(keypoints)
            if not knee_aligned:
                form_issues.append(knee_feedback)
                
            balanced, balance_feedback = self.check_balance(keypoints)
            if not balanced:
                form_issues.append(balance_feedback)
                
            good_posture, posture_feedback = self.check_posture(keypoints)
            if not good_posture:
                form_issues.append(posture_feedback)
            
            # Combine form issues
            good_form = len(form_issues) == 0
            
            current_time = time.time()
            
            # If not in correct form, provide feedback
            if not good_form:
                if current_time - self.last_message_time >= self.message_cooldown:
                    self.feedback = "Adjust your form"
                    self.detailed_feedback = " • " + "\n • ".join(form_issues)
                    self.feedback_color = (0, 120, 255)  # Orange for corrections (not harsh red)
                    self.last_message_time = current_time
                self.is_squatting = False
                self.squat_start_time = 0
                return
            
            # Handle squat transitions
            if current_level > 0:
                # Started or continuing a squat
                if not self.is_squatting or self.last_squat_level != current_level:
                    # New squat or changed level
                    self.is_squatting = True
                    self.squat_start_time = current_time
                    self.last_squat_level = current_level
                    
                    level_name = self.levels[current_level]['name']
                    self.feedback = f"Holding {level_name} Squat..."
                    self.detailed_feedback = "Excellent form!\nKeep holding steady."
                    self.feedback_color = (255, 255, 0)  # Yellow for in-progress
                elif self.is_squatting:
                    # Continuing same level squat
                    hold_duration = current_time - self.squat_start_time
                    
                    # Check if held long enough
                    if hold_duration >= self.hold_time_required:
                        # Complete the squat if we haven't given points for this hold yet
                        if self.last_squat_level == current_level:
                            # Award points (higher points for elderly accomplishments)
                            points = 15 * current_level
                            self.score += points
                            self.completed_squats[current_level] += 1
                            self.total_time_in_squat += hold_duration
                            
                            # Update feedback with encouragement for elderly users
                            level_name = self.levels[current_level]['name']
                            self.feedback = f"{level_name} Squat complete!"
                            self.detailed_feedback = f"+{points} points\nWonderful job!"
                            self.feedback_color = (0, 255, 0)  # Green for success
                            self.last_message_time = current_time
                            self.last_rep_time = current_time
                            
                            # Reset for next squat
                            self.is_squatting = False
                            self.last_squat_level = 0
            else:
                # Not actively squatting
                time_since_last_rep = current_time - self.last_rep_time
                if time_since_last_rep > 5.0 and time_since_last_rep < 6.0:
                    # Gentle reminder if they've been standing a while
                    self.feedback = "Ready for next squat when you are"
                    self.detailed_feedback = "Take your time.\nBend knees gently when ready."
                    self.feedback_color = (200, 200, 200)  # Light gray for gentle reminder
                self.is_squatting = False
                
        except Exception as e:
            print(f"Error updating squat: {e}")
    
    def draw_feedback(self, frame, position=(20, 120)):
        """Draw squat feedback on the frame with elderly-friendly format."""
        try:
            # Always show title
            cv2.putText(frame, "Squat Exercise", position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Main feedback - make larger and clearer for elderly users
            if self.feedback:
                pos_y = position[1] + 40
                cv2.putText(frame, self.feedback, (position[0], pos_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.feedback_color, 2)
                           
            # Detailed feedback
            if self.detailed_feedback:
                lines = self.detailed_feedback.split('\n')
                y_offset = position[1] + 80
                for line in lines:
                    cv2.putText(frame, line, (position[0], y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.feedback_color, 1)
                    y_offset += 25
            
            # If actively squatting, show hold timer with larger font
            if self.is_squatting:
                hold_time = time.time() - self.squat_start_time
                timer_text = f"Hold: {hold_time:.1f}s / {self.hold_time_required:.1f}s"
                y_offset = position[1] + 170
                
                # Color changes as time progresses (gentler transition for elderly)
                progress = min(1.0, hold_time / self.hold_time_required)
                timer_color = (
                    int(200 * (1 - progress)),  # R decreases (not too harsh)
                    int(200 * progress),       # G increases (not too bright)
                    int(100 * (1 - progress))  # B decreases (easier to see)
                )
                
                cv2.putText(frame, timer_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2)  # Larger font
                           
            # Show statistics - larger font and more spaced out
            y_offset = position[1] + 210
            total_squats = sum(self.completed_squats.values())
            cv2.putText(frame, f"Total Squats: {total_squats}", 
                       (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            y_offset += 30
            level_stats = f"L1:{self.completed_squats[1]} L2:{self.completed_squats[2]} L3:{self.completed_squats[3]}"
            cv2.putText(frame, level_stats, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                       
            # Safety reminder for elderly
            y_offset += 40
            safety_tip = "Remember: Go at your own pace"
            cv2.putText(frame, safety_tip, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 230, 255), 1)
                
        except Exception as e:
            print(f"Error drawing squat feedback: {e}")


# ------------------------------
# 7) Main
# ------------------------------
def main():
    try:
        # Tkinter popup with larger text and simpler instructions for elderly users
        root = tk.Tk()
        root.withdraw()
        
        # More readable instructions with larger font
        root.option_add('*Font', 'Arial 12')
        
        messagebox.showinfo("Physiotherapy Exercise Program", 
                          "Elderly-Friendly Exercise Program\n\n" +
                          "This program will guide you through various exercises:\n" +
                          "• Arm Raising - Gentle shoulder mobility\n" +
                          "• Side Stretches - Improve lateral flexibility\n" +
                          "• Chair Squats - Build leg strength\n\n" +
                          "CONTROLS:\n" +
                          "Press 1 - Arm Raising Exercise\n" +
                          "Press 2 - Side Stretch Exercise\n" +
                          "Press 3 - Chair Squat Exercise\n" +
                          "Press R - Restart current exercise\n" +
                          "Press Q - Quit program\n\n" +
                          "SAFETY TIPS:\n" +
                          "• Go at your own pace\n" +
                          "• Stop if you feel pain\n" +
                          "• Use a chair for support if needed")
        root.destroy()

        # Initialize game components
        pose_estimator = YoloPoseEstimator(model_path='yolov8n-pose.pt', conf=0.45)
        game_manager = GameManager(num_players=2)

        # Camera setup
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera.")
            return

        # Lower resolution for better performance but still readable for elderly
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        # Frame limiting - lower fps for more stable processing
        target_fps = 24
        frame_interval = 1.0 / target_fps
        prev_time = time.time()

        # Set initial window size larger for better visibility
        cv2.namedWindow("Physiotherapy Exercise Program", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Physiotherapy Exercise Program", 1024, 768)

        while True:
            # Frame rate control
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed < frame_interval:
                continue
            prev_time = current_time

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Pose detection
            keypoints_list = pose_estimator.estimate_pose(frame)

            # Get player colors for skeleton drawing
            player_colors = [p.color for p in game_manager.players]
            
            # Draw skeleton with player-specific colors
            pose_estimator.draw_skeleton(frame, keypoints_list, player_colors)

            # Process game logic
            game_manager.process_frame(frame, keypoints_list)
            
            # Animate floating text
            game_manager.attack_exercise.draw_messages(frame)

            # Display the frame
            cv2.imshow("Physiotherapy Exercise Program", frame)
            
            # Handle keyboard input - check with waitKey for longer duration for elderly users
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                game_manager.restart_game()
            elif key == ord('1'):
                print("Switching to Arm Raising exercise")
                game_manager.switch_exercise('attack')
            elif key == ord('2'):
                print("Switching to Side Stretch exercise")
                game_manager.switch_exercise('side_stretch')
            elif key == ord('3'):
                print("Switching to Chair Squat exercise")
                game_manager.switch_exercise('squat')

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing program and releasing resources...")
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import time
import math
import os
import random
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import dlib  # For facial landmark detection
import importlib.util

# ------------------------------
# FacialSafetyMonitor Class
# ------------------------------
class FacialSafetyMonitor:
    """Monitor facial expressions for signs of strain or discomfort during exercises."""
    
    def __init__(self):
        # Try to load dlib if available
        self.has_dlib = importlib.util.find_spec("dlib") is not None
        self.predictor = None
        self.detector = None
        
        # Load models if dlib is available
        if self.has_dlib:
            try:
                self.detector = dlib.get_frontal_face_detector()
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if os.path.exists(predictor_path):
                    self.predictor = dlib.shape_predictor(predictor_path)
            except Exception as e:
                print(f"Error loading facial detection models: {e}")
                self.has_dlib = False
        
        # Current facial expression state
        self.current_expression = "neutral"
        self.expression_confidence = 0.0
        self.expression_start_time = time.time()
        self.expression_colors = {
            "struggling": (30, 30, 220),  # Red
            "serious": (30, 220, 220),    # Yellow
            "enjoying": (30, 220, 30),    # Green
            "neutral": (200, 200, 200)    # Light gray
        }
        
        # Expression icons (simple Unicode symbols)
        self.expression_icons = {
            "struggling": "😣",  # Persevering face
            "serious": "😐",     # Neutral face
            "enjoying": "😊",    # Smiling face
            "neutral": "😶"      # Face without mouth
        }
        
        # Store expression history
        self.expression_history = []
        self.last_feedback_time = time.time()
        self.feedback_cooldown = 10  # seconds between feedback
        
        print("Facial safety monitoring system initialized.")
        if not self.has_dlib:
            print("Warning: dlib not available, using fallback detection")
    
    def analyze_expression(self, keypoints, frame):
        """
        Analyze facial expression using a combination of facial landmarks and pose estimation.
        Returns detected expression and confidence level.
        """
        # Default values
        expression = "neutral"
        confidence = 0.5
        
        # Primary method: Use dlib facial landmarks if available
        if self.has_dlib and self.predictor:
            try:
                # Convert frame to grayscale for dlib
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use nose position from keypoints to narrow search area
                nose_x, nose_y = keypoints[0]
                h, w = frame.shape[:2]
                
                # Define face search area around the nose
                face_search_left = max(0, int(nose_x - w/6))
                face_search_top = max(0, int(nose_y - h/6))
                face_search_right = min(w, int(nose_x + w/6))
                face_search_bottom = min(h, int(nose_y + h/6))
                
                # Create a region of interest for face detection
                roi = gray[face_search_top:face_search_bottom, face_search_left:face_search_right]
                
                # Detect faces in the region
                faces = self.detector(roi)
                
                if faces:
                    # Adjust face rectangle coordinates to the original frame
                    face_rect = faces[0]
                    face_rect = dlib.rectangle(
                        face_rect.left() + face_search_left,
                        face_rect.top() + face_search_top,
                        face_rect.right() + face_search_left,
                        face_rect.bottom() + face_search_top
                    )
                    
                    # Get facial landmarks
                    landmarks = self.predictor(gray, face_rect)
                    
                    # Analyze mouth shape
                    mouth_width = landmarks.part(54).x - landmarks.part(48).x
                    mouth_height = (landmarks.part(57).y - landmarks.part(51).y)
                    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
                    
                    # Analyze eyebrow position
                    left_eyebrow_y = landmarks.part(24).y
                    right_eyebrow_y = landmarks.part(19).y
                    eyebrow_baseline = (landmarks.part(36).y + landmarks.part(45).y) / 2
                    eyebrow_height = eyebrow_baseline - (left_eyebrow_y + right_eyebrow_y) / 2
                    
                    # Determine expression
                    if mouth_ratio > 0.5 or eyebrow_height < -5:  # Open mouth or furrowed brow
                        expression = "struggling"
                        confidence = min(1.0, max(0.7, mouth_ratio * 0.8 + abs(eyebrow_height) * 0.05))
                    elif mouth_ratio < 0.2 and eyebrow_height > 2:  # Smiling
                        expression = "enjoying"
                        confidence = min(1.0, 0.6 + eyebrow_height * 0.05)
                    elif abs(eyebrow_height) < 2 and mouth_ratio < 0.3:  # Neutral/focused
                        expression = "serious"
                        confidence = 0.7
            except Exception as e:
                print(f"Error in facial landmark analysis: {e}")
        
        # Fallback method: Use pose keypoints for basic expression detection
        if expression == "neutral" or confidence < 0.6:
            try:
                # Check head tilt (can indicate strain)
                if len(keypoints) >= 5:  # Assuming keypoints include face points
                    # Get nose, left ear, right ear positions
                    nose = keypoints[0]
                    left_ear = keypoints[3]
                    right_ear = keypoints[4]
                    
                    # Calculate head tilt
                    ear_y_diff = abs(left_ear[1] - right_ear[1])
                    ear_distance = ((left_ear[0] - right_ear[0])**2 + (left_ear[1] - right_ear[1])**2)**0.5
                    
                    if ear_distance > 0:
                        tilt_ratio = ear_y_diff / ear_distance
                        
                        if tilt_ratio > 0.3:  # Significant tilt
                            expression = "struggling"
                            confidence = min(1.0, 0.6 + tilt_ratio * 0.8)
                
                # Check arm positions (raised arms for lengthy periods can indicate strain)
                shoulders_y = (keypoints[5][1] + keypoints[6][1]) / 2 if len(keypoints) > 6 else 0
                arms_raised = False
                if len(keypoints) > 10:  # If we have arm keypoints
                    wrists_y = (keypoints[9][1] + keypoints[10][1]) / 2
                    if wrists_y < shoulders_y - 50:  # Arms are raised
                        arm_raise_duration = time.time() - self.expression_start_time
                        if arm_raise_duration > 20 and self.current_expression != "struggling":
                            expression = "struggling"
                            confidence = min(1.0, 0.6 + arm_raise_duration * 0.01)
            except Exception as e:
                print(f"Error in keypoint-based expression analysis: {e}")
        
        # Update expression if confidence is high enough or current is neutral
        if confidence > 0.65 or self.current_expression == "neutral":
            if expression != self.current_expression:
                self.expression_start_time = time.time()
                self.current_expression = expression
                self.expression_confidence = confidence
            elif confidence > self.expression_confidence:  # Update confidence if higher
                self.expression_confidence = confidence
        
        # Add to history
        timestamp = time.time()
        self.expression_history.append((timestamp, expression, confidence))
        
        # Prune old history
        self.expression_history = [entry for entry in self.expression_history 
                                if timestamp - entry[0] < 60]  # Keep last minute
        
        return self.current_expression, self.expression_confidence
    
    def get_feedback(self, expression):
        """Generate feedback based on facial expression."""
        current_time = time.time()
        
        # Respect cooldown unless expression is struggling
        if expression != "struggling" and current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
        
        feedback = None
        if expression == "struggling":
            # Feedback for signs of strain or discomfort
            feedback_options = [
                "Take a short rest if needed",
                "Remember to breathe deeply",
                "Listen to your body",
                "Go at your own pace",
                "It's okay to modify the exercise"
            ]
            feedback = random.choice(feedback_options)
            self.last_feedback_time = current_time
            
        elif expression == "serious":
            # Feedback for focused concentration
            feedback_options = [
                "Good focus!",
                "Nice form, keep it up",
                "You're doing well"
            ]
            if random.random() < 0.3:  # Occasional feedback
                feedback = random.choice(feedback_options)
                self.last_feedback_time = current_time
                
        elif expression == "enjoying":
            # Positive reinforcement
            feedback_options = [
                "Great job!",
                "Excellent work!",
                "You're making progress!",
                "Keep up the good work!"
            ]
            if random.random() < 0.5:  # More frequent positive feedback
                feedback = random.choice(feedback_options)
                self.last_feedback_time = current_time
        
        return feedback
    
    def draw_expression_status(self, frame, position=(50, 50)):
        """Draw current facial expression status with an icon and pulse effect."""
        try:
            x, y = position
            
            # Get current expression details
            expression = self.current_expression
            confidence = self.expression_confidence
            color = self.expression_colors.get(expression, (200, 200, 200))
            icon = self.expression_icons.get(expression, "😶")
            
            # Create pulsing effect for struggling expression
            if expression == "struggling":
                pulse = math.sin(time.time() * 5) * 0.2 + 0.8  # Pulse between 0.6 and 1.0
                color = tuple(int(c * pulse) for c in color)
            
            # Draw expression icon/emoji (using text as placeholder)
            cv2.putText(frame, icon, (x - 25, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw expression name and confidence
            status_text = f"{expression.capitalize()}"
            cv2.putText(frame, status_text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
        except Exception as e:
            print(f"Error drawing expression status: {e}")

# ------------------------------
# 1) Player Class
# ------------------------------
class Player:
    """
    Represents a player in the game with health points, attack power, and other attributes.
    """
    def __init__(self, player_id, max_hp=1000):
        self.id = player_id
        self.name = f"Player {player_id + 1}"
        self.max_hp = max_hp
        self.current_hp = max_hp
        
        # Attack and heal stats
        self.attack_power = 20
        self.heal_power = 0
        self.streak = 0
        self.streak_multiplier = 1.0
        
        # Set player color based on ID
        if player_id == 0:
            self.color = (30, 30, 220)  # Red for Player 1 (BGR)
        else:
            self.color = (30, 220, 30)  # Green for Player 2 (BGR)
        
        # Powerups
        self.active_powerups = {}
        
        # Comeback mechanics to help trailing players
        self.comeback_bonus = 1.0  # Multiplier for healing/reduced damage
        
        # Experience and leveling
        self.xp = 0
        self.level = 1
        self.xp_to_level = 100  # XP needed for first level up
        
        # Achievement tracking
        self.achievements = {
            "perfect_form": 0,
            "consecutive_hits": 0,
            "comeback_wins": 0,
            "exercises_completed": 0
        }
        
        # Animation state
        self.damage_animation_time = 0
        self.heal_animation_time = 0
        
        # Attack state for cooldown
        self.last_attack_time = 0
        self.attack_cooldown = 1.0  # seconds
    
    def take_damage(self, damage):
        """Apply damage to the player, considering comeback bonus"""
        # Apply comeback bonus for damage reduction (if trailing)
        if self.comeback_bonus > 1.0:
            damage = damage / (self.comeback_bonus * 0.75)  # Reduce damage based on comeback bonus
        
        # Calculate actual damage after modifiers
        actual_damage = min(self.current_hp, damage)
        self.current_hp -= actual_damage
        
        # Set damage animation
        self.damage_animation_time = time.time()
        
        # Reset streak if taking damage
        if self.streak > 1:
            self.streak = 0
        
        # Check if defeated
        is_defeated = self.current_hp <= 0
        if is_defeated:
            self.current_hp = 0
        
        return is_defeated
    
    def heal(self, amount):
        """Heal the player by the given amount, considering comeback bonus"""
        # Apply comeback bonus for increased healing (if trailing)
        if self.comeback_bonus > 1.0:
            amount = amount * self.comeback_bonus
        
        # Calculate actual healing after bonuses
        before_heal = self.current_hp
        self.current_hp = min(self.max_hp, self.current_hp + amount)
        actual_heal = self.current_hp - before_heal
        
        # Set heal animation
        self.heal_animation_time = time.time()
        
        # Award XP for healing
        self.add_xp(actual_heal * 0.5)
        
        return actual_heal
    
    def successful_attack(self, damage_dealt):
        """Update player stats after a successful attack"""
        # Increase streak counter
        self.streak += 1
        
        # Calculate streak multiplier (caps at 2.0)
        self.streak_multiplier = min(2.0, 1.0 + self.streak * 0.1)
        
        # Award XP based on damage dealt
        self.add_xp(damage_dealt * 0.75)
        
        # Update last attack time
        self.last_attack_time = time.time()
        
        # Update achievements
        self.achievements["consecutive_hits"] = max(self.achievements["consecutive_hits"], self.streak)
    
    def reset_attack(self):
        """Reset attack-related attributes"""
        self.streak = 0
        self.streak_multiplier = 1.0
    
    def add_xp(self, amount):
        """Add experience points and level up if necessary"""
        self.xp += amount
        
        # Check for level up
        if self.xp >= self.xp_to_level:
            self.level_up()
    
    def level_up(self):
        """Level up the player and increase stats"""
        self.level += 1
        
        # Increase attack power with each level
        self.attack_power += 5
        
        # Increase max HP with each level
        hp_increase = 100
        self.max_hp += hp_increase
        self.current_hp += hp_increase  # Also heal when leveling up
        
        # Increase XP requirement for next level (becomes progressively harder)
        self.xp_to_level = int(self.xp_to_level * 1.5)
        
        # Reset XP overflow
        self.xp = 0
    
    def add_powerup(self, powerup_type, duration=10):
        """Add a powerup to the player"""
        if powerup_type == "Shield":
            # Damage reduction
            self.active_powerups["Shield"] = {
                "end_time": time.time() + duration,
                "effect": 0.5  # 50% damage reduction
            }
        elif powerup_type == "Strength":
            # Increased attack
            self.active_powerups["Strength"] = {
                "end_time": time.time() + duration,
                "effect": 2.0  # Double attack power
            }
        elif powerup_type == "Regen":
            # Health regeneration
            self.active_powerups["Regen"] = {
                "end_time": time.time() + duration,
                "effect": 5  # HP per second
            }
    
    def update_powerups(self):
        """Update active powerups and apply effects"""
        current_time = time.time()
        expired_powerups = []
        
        for powerup, data in self.active_powerups.items():
            if current_time > data["end_time"]:
                expired_powerups.append(powerup)
            elif powerup == "Regen":
                # Apply regeneration effect
                self.heal(data["effect"] * 0.05)  # Scale down for smoother regen
        
        # Remove expired powerups
        for powerup in expired_powerups:
            del self.active_powerups[powerup]
    
    def update_comeback_bonus(self, opponent_health_percent):
        """Update comeback bonus based on difference in health percentages"""
        my_health_percent = self.current_hp / self.max_hp
        
        # If player is significantly behind (at least 30% difference)
        if my_health_percent < opponent_health_percent - 0.3:
            # Scale comeback bonus based on health difference
            health_diff = opponent_health_percent - my_health_percent
            self.comeback_bonus = 1.0 + min(1.0, health_diff * 1.5)  # Cap at 2.5x 
        else:
            # Reset bonus if not significantly behind
            self.comeback_bonus = 1.0
    
    def draw_position_indicator(self, frame):
        """Draw a small indicator at player's skeleton position"""
        if self.position[0] == 0 and self.position[1] == 0:
            return
        
        x, y = self.position
        
        # Draw name tag above player
        cv2.putText(frame, self.name, (x - 30, y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 1)

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
    
    Enhanced with simplified gamification:
    - Arm Raising: Attacks opponent (decreases their health)
    - Side Stretch: Heals yourself (increases your health)
    - Chair Squat: Heals yourself (increases your health)
    - Comeback mechanics for trailing players
    - Clean, minimalist interface with traditional health bars
    """
    def __init__(self, num_players=2):
        # Initialize players
        self.players = [Player(i, max_hp=1000) for i in range(num_players)]
        
        # Initialize exercise classes
        self.attack_exercise = AttackExercise(max_level=5, hp_per_level=4)
        self.side_stretch_left = SideStretch()
        self.side_stretch_right = SideStretch()
        self.squat_exercise = SquatExercise()
        
        # Initialize safety monitor
        self.safety_monitor = FacialSafetyMonitor()
        
        # Exercise selection
        self.current_exercise = 'attack'  # Set default exercise to 'attack'
        self.exercise_names = {
            'attack': 'Arm Raising (Attack)',
            'side_stretch': 'Side Stretch (Heal)',
            'squat': 'Chair Squat (Heal)'
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
        
        # Enhanced UI settings
        self.font_scale = 0.7
        self.font_thickness = 1
        self.colors = {
            'text': (255, 255, 255),           # White
            'background': (10, 10, 30),        # Deep blue-black
            'health_p1': (30, 30, 220),        # Red (BGR)
            'health_p2': (30, 220, 30),        # Green (BGR)
            'health_bg': (70, 70, 70),         # Dark gray
            'health_border': (200, 200, 200),  # Light gray
            'instruction': (200, 200, 200),    # Light gray
            'warning': (50, 150, 230),         # Soft orange
            'ui_accent': (255, 200, 40),       # Golden yellow
            'ui_highlight': (180, 230, 255),   # Light cyan
        }
        
        # Visual effects
        self.screen_flash = 0  # For attack/heal flash effects
        self.screen_flash_color = (0, 0, 0)
        self.screen_flash_start = 0
        
        # Animation timers
        self.heart_animation = 0
        self.heart_animation_start = 0
        self.heart_scale = 1.0
        self.health_bar_pulse = 0
        
        # Power-up spawn timing
        self.last_powerup_time = time.time()
        self.powerup_interval = 30  # seconds
        
        # Save current keypoints for animation effects
        self._current_keypoints_list = []
        
        # Initialize exercises for both sides
        self.side_stretch_left.set_stretch_side('left')
        self.side_stretch_right.set_stretch_side('right')
        
        # Add players to attack exercise
        for player in self.players:
            self.attack_exercise.add_player(player)
    
    def process_frame(self, frame, keypoints_list):
        """Process a frame and update game state based on the current exercise."""
        try:
            # Save keypoints for animation effects
            self._current_keypoints_list = keypoints_list
            
            # Update exercise timing
            current_time = time.time()
            elapsed = current_time - self.last_exercise_switch_time
            self.exercise_time[self.current_exercise] += elapsed
            self.last_exercise_switch_time = current_time
            
            # Process facial expressions for safety monitoring
            if len(keypoints_list) > 0:
                expression, confidence = self.safety_monitor.analyze_expression(keypoints_list[0], frame)
                
                # Get feedback if needed based on expression
                feedback = self.safety_monitor.get_feedback(expression)
                if feedback and expression == "struggling":
                    # Immediately show feedback for struggling users
                    self.attack_exercise._add_floating_text(
                        feedback,
                        int(keypoints_list[0][0][0]),  # nose x
                        int(keypoints_list[0][0][1] - 100),  # above nose
                        (50, 50, 255)  # red for urgency
                    )
                elif feedback and current_time % 15 < 0.1:  # Occasionally show for other expressions
                    self.attack_exercise._add_floating_text(
                        feedback,
                        int(keypoints_list[0][0][0]),  # nose x
                        int(keypoints_list[0][0][1] - 100),  # above nose
                        (50, 255, 50) if expression == "enjoying" else (255, 255, 50)  # green or yellow
                    )
            
            # Update comeback bonuses based on relative health
            self._update_comeback_mechanics()
            
            # Handle powerup spawning
            self._handle_powerups(current_time)
            
            # Process each player's movement based on the current exercise
            for i, player in enumerate(self.players):
                if i < len(keypoints_list):
                    if self.current_exercise == 'attack':
                        # OFFENSE: Arm Raising attacks opponent
                        attack_msg, target_id = self.attack_exercise.process_keypoints(i, keypoints_list[i])
                        
                        # Apply attack if one was triggered
                        if attack_msg and target_id is not None and target_id < len(self.players):
                            target_player = self.players[target_id]
                            attack_power = player.attack_power
                            
                            # Apply damage to target player
                            is_defeated = target_player.take_damage(attack_power)
                            
                            # Update attacker's streak and XP
                            player.successful_attack(attack_power)
                            
                            # Trigger heart animation
                            self.heart_animation = 1.0
                            self.heart_animation_start = current_time
                            
                            # Visual attack effect
                            self._trigger_screen_flash((0, 0, 200), 0.3)  # Red flash for attack
                            
                            # Check for game over
                            if is_defeated:
                                self.game_over = True
                                self.winner = player
                                if target_player.current_hp < target_player.max_hp * 0.2:
                                    player.achievements["comeback_wins"] += 1
                                
                        # Reset attack state if needed        
                        self.attack_exercise.finalize_if_needed(i)
                        
                    elif self.current_exercise == 'side_stretch':
                        # HEALING: Side Stretch heals player
                        if i == 0:  # Player 1 does left side stretch
                            self.side_stretch_left.update_stretch(keypoints_list[i])
                            # Apply healing if available
                            if self.side_stretch_left.heal_power > 0:
                                heal_amount = self._apply_healing(i, self.side_stretch_left.heal_power)
                                # Add achievement if it was a perfect form
                                if self.side_stretch_left.heal_power >= 15:  # Level 3 stretch
                                    player.achievements["perfect_form"] += 1
                                # Reset heal power after applying
                                self.side_stretch_left.heal_power = 0
                                
                        elif i == 1:  # Player 2 does right side stretch
                            self.side_stretch_right.update_stretch(keypoints_list[i])
                            # Apply healing if available
                            if self.side_stretch_right.heal_power > 0:
                                heal_amount = self._apply_healing(i, self.side_stretch_right.heal_power)
                                # Add achievement if it was a perfect form
                                if self.side_stretch_right.heal_power >= 15:  # Level 3 stretch
                                    player.achievements["perfect_form"] += 1
                                # Reset heal power after applying
                                self.side_stretch_right.heal_power = 0
                            
                    elif self.current_exercise == 'squat':
                        # HEALING: Chair Squat heals player
                        self.squat_exercise.update_squat(keypoints_list[i])
                        # Apply healing if available
                        if self.squat_exercise.heal_power > 0:
                            heal_amount = self._apply_healing(i, self.squat_exercise.heal_power)
                            # Add achievement if it was a perfect form
                            if self.squat_exercise.heal_power >= 24:  # Level 3 squat
                                player.achievements["perfect_form"] += 1
                            # Reset heal power after applying
                            self.squat_exercise.heal_power = 0
                    
                    # Update player powerups
                    player.update_powerups()

            # Draw UI elements
            self._draw_game_ui(frame)
            
            # Apply screen flash effects if active
            self._apply_screen_effects(frame, current_time)
            
        except Exception as e:
            print(f"Error in game processing: {e}")
    
    def _apply_healing(self, player_id, heal_amount):
        """Apply healing to a player with comeback bonus"""
        if player_id < len(self.players):
            player = self.players[player_id]
            
            # Apply the healing and get actual amount healed
            actual_heal = player.heal(heal_amount)
            
            # Trigger heart animation
            self.heart_animation = 1.0
            self.heart_animation_start = time.time()
            
            # Visual heal effect
            self._trigger_screen_flash((0, 200, 0), 0.3)  # Green flash for healing
            
            return actual_heal
        return 0
    
    def _update_comeback_mechanics(self):
        """Update comeback bonuses for players who are behind"""
        if len(self.players) < 2:
            return
            
        # Calculate health percentages
        p1_percent = self.players[0].current_hp / self.players[0].max_hp
        p2_percent = self.players[1].current_hp / self.players[1].max_hp
        
        # Update comeback bonuses for both players
        self.players[0].update_comeback_bonus(p2_percent)
        self.players[1].update_comeback_bonus(p1_percent)
    
    def _trigger_screen_flash(self, color, duration=0.2):
        """Trigger a screen flash effect"""
        self.screen_flash = 1.0
        self.screen_flash_color = color
        self.screen_flash_start = time.time()
        self.screen_flash_duration = duration
    
    def _apply_screen_effects(self, frame, current_time):
        """Apply active screen effects like flashes"""
        if self.screen_flash > 0:
            # Calculate flash intensity based on time
            elapsed = current_time - self.screen_flash_start
            if elapsed < self.screen_flash_duration:
                # Fade out effect
                intensity = 1.0 - (elapsed / self.screen_flash_duration)
                
                # Create overlay
                overlay = frame.copy()
                color = tuple(int(c * intensity * 0.5) for c in self.screen_flash_color)
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
                
                # Apply overlay with alpha
                alpha = 0.3 * intensity
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                self.screen_flash = 0
    
    def _handle_powerups(self, current_time):
        """Spawn and manage powerups for the player who's behind"""
        # Check if it's time to spawn a powerup
        if current_time - self.last_powerup_time > self.powerup_interval:
            # Give powerup to player with lower health
            if len(self.players) >= 2:
                p1_hp_percent = self.players[0].current_hp / self.players[0].max_hp
                p2_hp_percent = self.players[1].current_hp / self.players[1].max_hp
                
                if abs(p1_hp_percent - p2_hp_percent) > 0.2:  # At least 20% difference
                    # Give powerup to player with less health
                    target_player = 0 if p1_hp_percent < p2_hp_percent else 1
                    
                    # Random powerup: Shield, Strength, or Regen
                    powerup_types = ["Shield", "Strength", "Regen"]
                    import random
                    powerup = random.choice(powerup_types)
                    
                    # Apply powerup
                    self.players[target_player].add_powerup(powerup, duration=15)
                    
                    # Add a notification
                    self.attack_exercise._add_floating_text(
                        f"{self.players[target_player].name} got {powerup} boost!",
                        frame.shape[1] // 2,
                        frame.shape[0] // 2 - 100,
                        (0, 255, 255)  # Yellow
                    )
                    
                    # Reset timer
                    self.last_powerup_time = current_time
    
    # ---- UI Drawing Helper Methods ----
    
    def _draw_heart_outline(self, frame, x, y, scale=1.0, alpha=1.0):
        """Draw heart outline for Player 1 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        heart_shape = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        heart_shape[:, 0] += x
        heart_shape[:, 1] += y
        
        # Draw outline with darker color
        color = tuple(int(c * alpha) for c in self.colors['health_border'])
        cv2.polylines(frame, [heart_shape], True, color, 2)
    
    def _draw_heart_filled(self, frame, x, y, scale=1.0, alpha=1.0, pulse=0):
        """Draw filled heart for Player 1 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Apply pulse effect
        pulse_scale = 1.0 + 0.1 * pulse
        w = int(w * pulse_scale)
        h = int(h * pulse_scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        heart_shape = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        heart_shape[:, 0] += x
        heart_shape[:, 1] += y
        
        # Draw filled heart
        color = tuple(int(c * alpha) for c in self.colors['health_p1'])
        cv2.fillPoly(frame, [heart_shape], color)
        
        # Draw outline for better visibility
        outline_color = tuple(int(min(c + 50, 255) * alpha) for c in self.colors['health_p1'])
        cv2.polylines(frame, [heart_shape], True, outline_color, 1)
    
    def _draw_heart_half(self, frame, x, y, scale=1.0, alpha=1.0):
        """Draw half-filled heart for Player 1 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        full_heart = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Half heart (left side)
        half_heart = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        full_heart[:, 0] += x
        full_heart[:, 1] += y
        half_heart[:, 0] += x
        half_heart[:, 1] += y
        
        # Draw outline of full heart
        outline_color = tuple(int(c * alpha) for c in self.colors['health_border'])
        cv2.polylines(frame, [full_heart], True, outline_color, 2)
        
        # Fill half of the heart
        color = tuple(int(c * alpha) for c in self.colors['health_p1'])
        cv2.fillPoly(frame, [half_heart], color)
    
    def _draw_player1_hearts(self, frame, player):
        """Draw Minecraft-style hearts for Player 1"""
        # Position at bottom left
        h, w = frame.shape[:2]
        base_x, base_y = 50, h - 50
        
        # Draw health bar background panel
        panel_height = 40
        panel_width = 320
        cv2.rectangle(frame, (10, base_y - 20), (10 + panel_width, base_y + panel_height), 
                     (0, 0, 0, 128), -1)
        
        # Draw player name
        cv2.putText(frame, player.name, (20, base_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['health_p1'], 1)
        
        # Calculate hearts
        hp_percent = player.current_hp / player.max_hp
        max_hearts = 10
        full_hearts = int(hp_percent * max_hearts)
        half_heart = (hp_percent * max_hearts) - full_hearts > 0.5
        
        # Scale and animation effects
        heart_scale = 1.0
        anim_pulse = 0
        
        # Apply animation if recent health change
        current_time = time.time()
        if self.heart_animation > 0:
            elapsed = current_time - self.heart_animation_start
            if elapsed < 0.5:
                # Heart animation effect
                anim_pulse = math.sin(elapsed * 20) * (1.0 - elapsed * 2)
                heart_scale = 1.0 + 0.2 * anim_pulse
            else:
                self.heart_animation = 0
        
        # Draw hearts with spacing
        heart_spacing = 25
        for i in range(max_hearts):
            x = base_x + i * heart_spacing
            
            # Heart animation for this specific heart
            this_heart_pulse = 0
            if i == full_hearts - 1 or (i == full_hearts and half_heart):
                this_heart_pulse = anim_pulse
                
            # Draw heart outlines first (all containers)
            self._draw_heart_outline(frame, x, base_y)
            
            # Draw filled hearts
            if i < full_hearts:
                self._draw_heart_filled(frame, x, base_y, pulse=this_heart_pulse)
            elif i == full_hearts and half_heart:
                self._draw_heart_half(frame, x, base_y)
        
        # Draw health percentage
        health_text = f"{int(hp_percent * 100)}%"
        cv2.putText(frame, health_text, (base_x + max_hearts * heart_spacing + 10, base_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['health_p1'], 1)
    
    def _draw_player2_bar(self, frame, player):
        """Draw Minecraft-style hearts for Player 2 (bottom right)"""
        # Position at bottom right
        h, w = frame.shape[:2]
        base_x, base_y = w - 300, h - 50
        
        # Draw health bar background panel
        panel_height = 40
        panel_width = 320
        cv2.rectangle(frame, (w - 10 - panel_width, base_y - 20), 
                     (w - 10, base_y + panel_height), (0, 0, 0, 128), -1)
        
        # Draw player name (right-aligned)
        name_size = cv2.getTextSize(player.name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(frame, player.name, (w - 20 - name_size[0], base_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['health_p2'], 1)
        
        # Calculate hearts
        hp_percent = player.current_hp / player.max_hp
        max_hearts = 10
        full_hearts = int(hp_percent * max_hearts)
        half_heart = (hp_percent * max_hearts) - full_hearts > 0.5
        
        # Scale and animation effects
        heart_scale = 1.0
        anim_pulse = 0
        
        # Apply animation if recent health change
        current_time = time.time()
        if self.heart_animation > 0:
            elapsed = current_time - self.heart_animation_start
            if elapsed < 0.5:
                # Heart animation effect
                anim_pulse = math.sin(elapsed * 20) * (1.0 - elapsed * 2)
                heart_scale = 1.0 + 0.2 * anim_pulse
            else:
                self.heart_animation = 0
        
        # Draw hearts with spacing - right to left for Player 2
        heart_spacing = 25
        for i in range(max_hearts):
            # Position from right to left
            x = base_x - i * heart_spacing
            
            # Heart animation for this specific heart
            this_heart_pulse = 0
            if i == max_hearts - full_hearts or (i == max_hearts - full_hearts - 1 and half_heart):
                this_heart_pulse = anim_pulse
                
            # Draw heart outlines first (all containers)
            self._draw_p2_heart_outline(frame, x, base_y)
            
            # Draw filled hearts
            if i < (max_hearts - full_hearts):
                # Empty hearts
                pass
            elif i == (max_hearts - full_hearts) and half_heart:
                # Half-filled heart
                self._draw_p2_heart_half(frame, x, base_y)
            else:
                # Fully filled hearts
                self._draw_p2_heart_filled(frame, x, base_y, pulse=this_heart_pulse)
        
        # Draw health percentage
        health_text = f"{int(hp_percent * 100)}%"
        cv2.putText(frame, health_text, (base_x - max_hearts * heart_spacing - 40, base_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['health_p2'], 1)
    
    def _draw_p2_heart_outline(self, frame, x, y, scale=1.0, alpha=1.0):
        """Draw heart outline for Player 2 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        heart_shape = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        heart_shape[:, 0] += x
        heart_shape[:, 1] += y
        
        # Draw outline with darker color
        color = tuple(int(c * alpha) for c in self.colors['health_border'])
        cv2.polylines(frame, [heart_shape], True, color, 2)
    
    def _draw_p2_heart_filled(self, frame, x, y, scale=1.0, alpha=1.0, pulse=0):
        """Draw filled heart for Player 2 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Apply pulse effect
        pulse_scale = 1.0 + 0.1 * pulse
        w = int(w * pulse_scale)
        h = int(h * pulse_scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        heart_shape = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        heart_shape[:, 0] += x
        heart_shape[:, 1] += y
        
        # Draw filled heart with player 2 color
        color = tuple(int(c * alpha) for c in self.colors['health_p2'])
        cv2.fillPoly(frame, [heart_shape], color)
        
        # Draw outline for better visibility
        outline_color = tuple(int(min(c + 50, 255) * alpha) for c in self.colors['health_p2'])
        cv2.polylines(frame, [heart_shape], True, outline_color, 1)
    
    def _draw_p2_heart_half(self, frame, x, y, scale=1.0, alpha=1.0):
        """Draw half-filled heart for Player 2 health"""
        heart_width, heart_height = 20, 20
        w = int(heart_width * scale)
        h = int(heart_height * scale)
        
        # Adjusted for center point
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Heart shape points
        full_heart = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [3*w/4, 0], 
            [w, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Half heart (left side)
        half_heart = np.array([
            [0, h/4], 
            [w/4, 0], 
            [w/2, h/4], 
            [w/2, h], 
            [0, h/4]
        ], dtype=np.int32)
        
        # Adjust points to position
        full_heart[:, 0] += x
        full_heart[:, 1] += y
        half_heart[:, 0] += x
        half_heart[:, 1] += y
        
        # Draw outline of full heart
        outline_color = tuple(int(c * alpha) for c in self.colors['health_border'])
        cv2.polylines(frame, [full_heart], True, outline_color, 2)
        
        # Fill half of the heart with player 2 color
        color = tuple(int(c * alpha) for c in self.colors['health_p2'])
        cv2.fillPoly(frame, [half_heart], color)
    
    def _draw_game_ui(self, frame):
        """Draw enhanced game UI elements"""
        try:
            # Draw player health indicators
            if len(self.players) >= 1:
                self._draw_player1_hearts(frame, self.players[0])
            if len(self.players) >= 2:
                self._draw_player2_bar(frame, self.players[1])
            
            # Draw session time in top-left
            session_time = time.time() - self.session_start_time
            minutes, seconds = divmod(int(session_time), 60)
            time_text = f"Time: {minutes:02d}:{seconds:02d}"
            
            # Draw semi-transparent header bar
            header_height = 30
            cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), 
                         (0, 0, 0, 180), -1)
            
            # Draw timer
            cv2.putText(frame, time_text, (10, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            
            # Draw current exercise name in top center
            if self.current_exercise:
                exercise_name = self.exercise_names.get(self.current_exercise, self.current_exercise.capitalize())
                # Center text
                text_size = cv2.getTextSize(exercise_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(frame, exercise_name, (text_x, 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
            
            # Draw facial expression status in a more compact form
            self.safety_monitor.draw_expression_status(frame, (frame.shape[1] - 160, 22))
            
            # Draw small instruction bar at bottom
            control_height = 25
            control_y = frame.shape[0] - control_height
            
            # Draw semi-transparent background for controls
            control_bg = frame.copy()
            cv2.rectangle(control_bg, (0, control_y), 
                         (frame.shape[1], frame.shape[0]), 
                         self.colors['background'], -1)
            cv2.addWeighted(control_bg, 0.6, frame, 0.4, 0, frame)
            
            # Draw controls with minimal text
            controls_text = "1=Attack | 2=Stretch | 3=Squat | r=Reset | q=Quit"
            text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, controls_text, (text_x, control_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['instruction'], 1)

            # Draw exercise-specific UI elements
            if self.current_exercise == 'side_stretch':
                # Display stretch information for both players
                if len(self.players) >= 1:
                    # Left side stretch feedback (Player 1)
                    self.side_stretch_left.draw_feedback(frame, position=(20, 120))
                
                if len(self.players) >= 2:
                    # Right side stretch feedback (Player 2)
                    self.side_stretch_right.draw_feedback(frame, position=(frame.shape[1] - 300, 120))
            
            elif self.current_exercise == 'squat':
                # Show squat exercise feedback
                self.squat_exercise.draw_feedback(frame, position=(30, 120))

            # Draw game state if game is over
            if self.game_over and self.winner:
                # Draw semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw winner text
                winner_text = f"{self.winner.name} WINS!"
                text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = frame.shape[0] // 2
                cv2.putText(frame, winner_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.winner.color, 2)
                
                # Instruction to restart
                restart_text = "Press 'r' to restart"
                text_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(frame, restart_text, (text_x, text_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['instruction'], 1)
                            
            # Draw floating messages
            self.attack_exercise.draw_messages(frame)
                
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
        
        # Reset powerup timer
        self.last_powerup_time = time.time()
            
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
    
    Restores health to the player when performed correctly.
    """
    def __init__(self):
        # Angles for different levels (in degrees)
        self.levels = {1: 15, 2: 30, 3: 45}  
        self.current_level = 0
        self.score = 0
        self.stretch_side = None  # 'left' or 'right'
        self.heal_power = 0  # How much health to restore
        
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
                self.heal_power = 0
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
                            # Set healing power based on level (5, 10, or 15 health)
                            self.heal_power = 5 * current_level
                            points = 10 * current_level
                            self.score += points
                            self.completed_stretches[current_level] += 1
                            
                            # Update feedback
                            self.feedback = f"Great stretch! +{self.heal_power} Health"
                            self.feedback_color = (0, 255, 0)  # Green for success
                            self.last_message_time = current_time
                            
                            # Reset for next stretch
                            self.is_stretching = False
                            self.last_stretch_level = 0
            else:
                # Not actively stretching
                self.is_stretching = False
                self.heal_power = 0
                
        except Exception as e:
            print(f"Error updating stretch: {e}")
            
    def draw_feedback(self, frame, position=(20, 120)):
        """Draw stretch feedback on the frame."""
        try:
            # Draw title
            cv2.putText(frame, "Side Stretch - Heals You", position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            # Draw the feedback text
            if self.feedback:
                y_pos = position[1] + 40
                cv2.putText(frame, self.feedback, (position[0], y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.feedback_color, 2)
                       
            # Draw stretch side indicator
            side_text = f"Stretch Side: {self.stretch_side.capitalize()}" if self.stretch_side else ""
            if side_text:
                y_offset = position[1] + 80
                cv2.putText(frame, side_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # If actively stretching, show hold timer
            if self.is_stretching:
                hold_time = time.time() - self.stretch_start_time
                timer_text = f"Hold: {hold_time:.1f}s / {self.hold_time_required:.1f}s"
                y_offset = position[1] + 120
                
                # Color changes as time progresses
                progress = min(1.0, hold_time / self.hold_time_required)
                timer_color = (
                    int(255 * (1 - progress)),  # R decreases
                    int(255 * progress),       # G increases
                    0                          # B stays 0
                )
                
                cv2.putText(frame, timer_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
                
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
    
    Restores health to the player when performed correctly.
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
        self.heal_power = 0  # How much health to restore
        
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
                self.heal_power = 0
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
                    self.detailed_feedback = "Good form!"
                    self.feedback_color = (255, 255, 0)  # Yellow for in-progress
                elif self.is_squatting:
                    # Continuing same level squat
                    hold_duration = current_time - self.squat_start_time
                    
                    # Check if held long enough
                    if hold_duration >= self.hold_time_required:
                        # Complete the squat if we haven't given points for this hold yet
                        if self.last_squat_level == current_level:
                            # Set healing power (8, 16, or 24 health)
                            self.heal_power = 8 * current_level
                            points = 15 * current_level
                            self.score += points
                            self.completed_squats[current_level] += 1
                            self.total_time_in_squat += hold_duration
                            
                            # Update feedback with encouragement for elderly users
                            level_name = self.levels[current_level]['name']
                            self.feedback = f"Great job! +{self.heal_power} Health"
                            self.detailed_feedback = "Well done!"
                            self.feedback_color = (0, 255, 0)  # Green for success
                            self.last_message_time = current_time
                            self.last_rep_time = current_time
                            
                            # Reset for next squat
                            self.is_squatting = False
                            self.last_squat_level = 0
            else:
                # Not actively squatting
                self.heal_power = 0
                time_since_last_rep = current_time - self.last_rep_time
                if time_since_last_rep > 5.0 and time_since_last_rep < 6.0:
                    # Gentle reminder if they've been standing a while
                    self.feedback = "Ready for next squat when you are"
                    self.detailed_feedback = "Bend knees when ready"
                    self.feedback_color = (200, 200, 200)  # Light gray for gentle reminder
                self.is_squatting = False
                
        except Exception as e:
            print(f"Error updating squat: {e}")
    
    def draw_feedback(self, frame, position=(20, 120)):
        """Draw squat feedback on the frame with elderly-friendly format."""
        try:
            # Always show title
            cv2.putText(frame, "Chair Squat - Heals You", position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Main feedback - make larger and clearer for elderly users
            if self.feedback:
                pos_y = position[1] + 40
                cv2.putText(frame, self.feedback, (position[0], pos_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.feedback_color, 2)
                           
            # Simplified detailed feedback
            if self.detailed_feedback:
                y_offset = position[1] + 80
                cv2.putText(frame, self.detailed_feedback, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.feedback_color, 1)
            
            # If actively squatting, show hold timer with larger font
            if self.is_squatting:
                hold_time = time.time() - self.squat_start_time
                timer_text = f"Hold: {hold_time:.1f}s / {self.hold_time_required:.1f}s"
                y_offset = position[1] + 120
                
                # Color changes as time progresses (gentler transition for elderly)
                progress = min(1.0, hold_time / self.hold_time_required)
                timer_color = (
                    int(200 * (1 - progress)),  # R decreases (not too harsh)
                    int(200 * progress),       # G increases (not too bright)
                    int(100 * (1 - progress))  # B decreases (easier to see)
                )
                
                cv2.putText(frame, timer_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2)  # Larger font
                           
            # Safety reminder for elderly
            y_offset = position[1] + 180
            safety_tip = "Remember: Go at your own pace"
            cv2.putText(frame, safety_tip, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 230, 255), 1)
                
        except Exception as e:
            print(f"Error drawing squat feedback: {e}")


# ------------------------------
# 7) Main
# ------------------------------
def main():
    # Ensure necessary data files are available
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    # Check if facial landmark predictor file exists
    if not os.path.exists(predictor_path):
        # Show popup with download instructions
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        message = (
            "The facial landmark predictor file is needed for safety monitoring.\n\n"
            "Please download it from:\n"
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n\n"
            "Extract the file to the same folder as this program.\n"
            "The program will continue, but facial safety monitoring will be limited."
        )
        
        messagebox.showinfo("Download Required", message)
    
    # Create a large popup message with clear instructions
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Detailed message for elderly users with clear instructions
    welcome_message = (
        "🏋️‍♀️ Physiotherapy Exercise Program 🏋️‍♀️\n\n"
        "Welcome to your exercise session!\n\n"
        "This program offers three exercises:\n"
        "1. Arm Raising (Buddha Clap) - Raise your arms to attack\n"
        "2. Side Stretch - Stretch to the side to heal yourself\n"
        "3. Chair Squat - Perform gentle squats to heal yourself\n\n"
        "Controls:\n"
        "- Press 1 for Arm Raising exercise\n"
        "- Press 2 for Side Stretch exercise\n"
        "- Press 3 for Chair Squat exercise\n"
        "- Press R to reset the session\n"
        "- Press Q to quit the program\n\n"
        "Safety First:\n"
        "• Go at your own pace\n"
        "• Stop if you feel any pain\n"
        "• The program will monitor your facial expressions for safety\n\n"
        "The enhanced interface includes:\n"
        "• Player 1: Minecraft-style health hearts\n"
        "• Player 2: Traditional health bar with visual effects\n"
        "• Animated health changes and feedback\n\n"
        "Enjoy your exercise session!"
    )
    
    messagebox.showinfo("Physiotherapy Exercise Program", welcome_message)
    
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    # Set target frame rate for more stable processing
    target_fps = 24
    
    # Initialize skeletal detection model
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        messagebox.showerror("Error", "Failed to load YOLO model. Please check installation.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Initialize game manager
    game_manager = GameManager(num_players=2)
    
    # Processing loop
    while True:
        # Timer to maintain consistent frame rate
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run detection model
        results = model.predict(frame, verbose=False)
        
        # Get keypoints
        keypoints_list = []
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            if len(keypoints) > 0:
                keypoints_list.append(keypoints[0])
        
        # Update game state with keypoints
        game_manager.process_frame(frame, keypoints_list)
        
        # Display frame
        cv2.imshow('Physiotherapy Exercise', frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            game_manager.restart_game()
        # Switch between exercises
        elif key == ord('1'):
            game_manager.switch_exercise('attack')
        elif key == ord('2'):
            game_manager.switch_exercise('side_stretch')
        elif key == ord('3'):
            game_manager.switch_exercise('squat')
            
        # Calculate delay to maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = max(1.0/target_fps - elapsed, 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

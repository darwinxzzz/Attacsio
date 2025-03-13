# Exercise Game with Safety Monitoring

An interactive exercise game designed especially for elderly users, combining physical activity with game elements and safety monitoring features.

## Features

- **Three Exercise Types**:
  - **Arm Raising (Attack)**: Raise your arms to damage your opponent
  - **Side Stretch (Heal)**: Stretch to the side to heal yourself
  - **Chair Squat (Heal)**: Perform gentle squats to heal yourself

- **Safety Monitoring**:
  - **Facial Expression Analysis**: Detects if users are struggling, serious, or enjoying the exercise
  - **Adaptive Feedback**: Provides personalized guidance based on facial expressions
  - **Safety Alerts**: Offers encouragement and safety tips when signs of strain are detected

- **Elderly-Friendly Design**:
  - Large, high-contrast text
  - Simple controls
  - Gentle exercise movements
  - Clear feedback

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the facial landmark predictor:
   - Go to: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Download and extract the file
   - Place `shape_predictor_68_face_landmarks.dat` in the same directory as the script

> **Note**: If the facial landmark predictor is not found, the program will still run with a simplified expression detection method.

## Usage

1. Run the main script:
   ```
   python script/main.py
   ```

2. Control the game with keyboard:
   - **1**: Switch to Arm Raising (Attack)
   - **2**: Switch to Side Stretch (Heal)
   - **3**: Switch to Chair Squat (Heal)
   - **r**: Restart the game
   - **q**: Quit the program

3. Exercise Instructions:
   - **Arm Raising**: Raise your arms from hips to overhead to attack opponent
   - **Side Stretch**: Lean to the side and hold to heal yourself
   - **Chair Squat**: Perform gentle knee bends to heal yourself
   
## Safety Features

The facial expression monitoring system detects:

- **Signs of Struggling**: Provides immediate feedback to slow down or rest
- **Focused Concentration**: Offers encouragement to maintain good form
- **Enjoyment**: Gives positive reinforcement

All exercise monitoring includes safety checks for proper form and alignment to prevent injury.

## Requirements

- Python 3.7+
- Webcam
- Sufficient lighting for good pose detection
- OpenCV
- Ultralytics YOLO
- dlib (recommended for better facial expression detection)

## For Caregivers

This program includes safety features specifically designed for elderly users:

1. Form validation to prevent improper movements
2. Facial expression monitoring to detect signs of strain
3. Adaptive difficulty based on user capability
4. Encouragement and safety reminders throughout the session

Monitor the "Current Mood" indicator in the top-right corner to see how the user is responding to the exercises. 
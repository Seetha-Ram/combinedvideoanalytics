import streamlit as st
import cv2
import os
from pathlib import Path
import tempfile
import numpy as np
import mediapipe as mp  # Add this import statement
import math

st.title('Video Difference Calculator and Pose Estimation App')

# Video Difference Calculator Section
st.header('Video Difference Calculator')

# Define upload folder for video files
upload_folder = 'uploads'

# Create upload folder if it doesn't exist
Path(upload_folder).mkdir(parents=True, exist_ok=True)

uploaded_files = st.file_uploader("Upload two video files", type=["mp4"], key="video_uploader", accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) == 2:
    file1, file2 = uploaded_files

    # Save uploaded files to the upload folder
    video1_path = os.path.join(upload_folder, file1.name)
    video2_path = os.path.join(upload_folder, file2.name)

    with open(video1_path, 'wb') as f1, open(video2_path, 'wb') as f2:
        f1.write(file1.read())
        f2.write(file2.read())

    # Continue with video processing as before
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    similarity_threshold = 0.95  # Set a threshold for similarity (adjust as needed)

    while True:
        success1, frame1 = video1.read()
        success2, frame2 = video2.read()

        if not success1 or not success2:
            break

        difference = cv2.absdiff(frame1, frame2)

        # Compute the similarity score
        similarity_score = np.sum(np.abs(frame1 - frame2)) / (frame1.size * frame1.itemsize)

        # Check similarity based on the threshold
        if similarity_score > similarity_threshold:
            st.success(f"Video similarity: {similarity_score:.2f}")
        else:
            st.warning(f"Video similarity: {similarity_score:.2f}. It does not meet the threshold.")

        # Display the processed frame
        st.image(difference, channels="RGB", use_column_width=True)

    video1.release()
    video2.release()
else:
    st.warning("Please upload two video files.")

# Pose Estimation App Section
st.header('Pose Estimation App')

# Upload a video file
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file:
    # Convert the uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file.close()
    video_path = temp_file.name

    # Display uploaded video
    st.video(video_path)

    # Initialize MediaPipe Pose solution
    mp_pose_holistic = mp.solutions.holistic

    def calculate_angle(a, b, c):
        # Calculate the angle between three points (in degrees)
        angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def process_frame(frame):
        # Load the Holistic model from MediaPipe
        with mp_pose_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Convert frame to RGB format (required by MediaPipe)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Perform Pose estimation on the frame
            results = holistic.process(frame)

            # Convert frame back to RGB format for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw landmarks and calculate angles
            if results.pose_landmarks:
                # Extract key points
                landmarks = results.pose_landmarks.landmark

                # Calculate angles between body parts
                left_shoulder = landmarks[mp_pose_holistic.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose_holistic.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose_holistic.PoseLandmark.LEFT_WRIST]

                right_shoulder = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_WRIST]

                neck = landmarks[mp_pose_holistic.PoseLandmark.NOSE]

                left_hip = landmarks[mp_pose_holistic.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose_holistic.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose_holistic.PoseLandmark.LEFT_ANKLE]

                right_hip = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_HIP]
                right_knee = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_KNEE]
                right_ankle = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_ANKLE]

                # Calculate the angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                neck_angle = calculate_angle(left_shoulder, neck, right_shoulder)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Draw skeletal lines
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks)
                draw_skeletal_lines(frame, results.pose_landmarks)

                # Display the angles on the frame
                cv2.putText(frame, f"Left Elbow Angle: {left_elbow_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Right Elbow Angle: {right_elbow_angle:.2f} degrees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Neck Angle: {neck_angle:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Left Knee Angle: {left_knee_angle:.2f} degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Right Knee Angle: {right_knee_angle:.2f} degrees", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame  # Return the processed frame

    def draw_skeletal_lines(frame, landmarks):
        # Define connections for drawing skeletal lines
        connections = mp_pose_holistic.POSE_CONNECTIONS

        # Loop through the connections and draw lines
        for connection in connections:
            start_point = connection[0]
            end_point = connection[1]

            # Get the landmark points
            start_landmark = landmarks.landmark[start_point]
            end_landmark = landmarks.landmark[end_point]

            # Convert landmark positions to pixel coordinates
            x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
            x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

            # Draw the skeletal lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Process the video and apply pose estimation
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the processed frame
        st.image(processed_frame, channels="RGB", use_column_width=True)

    cap.release()

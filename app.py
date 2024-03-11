import streamlit as st
import cv2
import os
import numpy as np
import tempfile
import mediapipe as mp
import math
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = '123'  # Change this to a secret key for session management

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        processing_option = request.form.get('processing_option')
        if processing_option == 'Video Similarity':
            return render_template('video_processing.html', processing_option='Video Similarity')
        elif processing_option == 'Pose Estimation':
            return render_template('video_processing.html', processing_option='Pose Estimation')
    return render_template('index.html')


# Set up SQLite database
conn = sqlite3.connect('user_database.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL,
        email TEXT NOT NULL
    )
''')
conn.commit()

UPLOAD_FOLDER = 'uploads'
DIFFERENCE_VIDEO_PATH = 'static/difference.mp4'

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app for video similarity and pose estimation
def video_similarity_app():
    st.subheader('Video Similarity')

    # Upload two videos
    file1 = st.file_uploader('Upload first video (.mp4)', type=['mp4'])
    file2 = st.file_uploader('Upload second video (.mp4)', type=['mp4'])

    if file1 and file2:
        # Process the videos
        process_button = st.button('Process Videos')
        if process_button:
            process_videos(file1, file2)
            st.success('Videos processed successfully!')

            # Display the processed video
            st.video(DIFFERENCE_VIDEO_PATH, format="video/mp4")

def pose_estimation_app():
    st.subheader('Pose Estimation')

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

        with mp_pose_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame = process_frame(frame, holistic)

                # Display the processed frame
                st.image(processed_frame, channels="RGB", use_column_width=True)

            # Release the video capture object
            cap.release()

def process_frame(frame, holistic):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose_holistic.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose_holistic.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose_holistic.PoseLandmark.LEFT_WRIST]

        # Add more landmark points as needed

        # Calculate the angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Add more angle calculations as needed

        # Draw skeletal lines
        mp.solutions.drawing_utils.draw_landmarks(frame_rgb, results.pose_landmarks)
        draw_skeletal_lines(frame_rgb, results.pose_landmarks)

        # Display the angles on the frame
        cv2.putText(frame_rgb, f"Left Elbow Angle: {left_elbow_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add more angle display statements as needed

    return frame_rgb

# Flask routes
@app.route('/flask', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        login_option = request.form.get('login_option')
        if login_option == 'Login':
            return login()
        elif login_option == 'Signup':
            return signup()

    return render_template('index.html')

def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if the entered credentials exist in the database
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()

    if user:
        session['user'] = username
        return redirect(url_for('video_processing'))
    else:
        return render_template('index.html', error="Invalid username or password. Please try again.")

def signup():
    new_username = request.form.get('new_username')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    new_email = request.form.get('new_email')

    if new_password == confirm_password:
        # Insert user details into the database
        cursor.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (new_username, new_password, new_email))
        conn.commit()

        return render_template('index.html', success="Sign up successful! You can now log in.")
    else:
        return render_template('index.html', error="Passwords do not match. Please try again.")


@app.route('/flask/video_processing', methods=['GET', 'POST'])
def video_processing():
    if 'user' in session:
        if request.method == 'POST':
            processing_option = request.form.get('processing_option')
            if processing_option == 'Video Similarity':
                return render_template('video_processing.html', username=session['user'], processing_option='Video Similarity')
            elif processing_option == 'Pose Estimation':
                return render_template('video_processing.html', username=session['user'], processing_option='Pose Estimation')

        return render_template('video_processing.html', username=session['user'])
    else:
        return redirect(url_for('index'))

# Run the Streamlit app at the root path '/'
st.set_page_config(page_title="Streamlit App", page_icon="🎥")
st.title('Streamlit Video Processing App')
st.sidebar.title('Navigation')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)  # Use a different port for Flask

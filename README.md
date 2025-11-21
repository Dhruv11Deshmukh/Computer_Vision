Face Recognition

This project implements a complete face recognition pipeline using classical computer vision techniques. It allows the user to register faces, detect multiple faces in real time, recognize known individuals, and simulate a face unlock mechanism. The system uses Local Binary Patterns (LBP) for feature extraction and Haar Cascade for face detection, making it lightweight and fast without needing deep learning models or GPU support.

Overview

The project consists of three main phases. The first phase is face registration, where a user’s face is captured through the webcam, converted to grayscale, processed using LBP, and stored as a feature vector along with the person’s name in a pickle file. The second phase is live multi-face detection and recognition. The webcam detects all faces in the frame, extracts LBP histograms for each face, compares them with stored embeddings using Euclidean distance, and assigns the closest match. The system also visualizes key facial texture regions based on the strongest LBP features. The third phase is a face unlock simulation. When a registered user appears in front of the camera and the match distance is below a defined threshold, the system switches from LOCKED to UNLOCKED for a few seconds, imitating phone-style face unlock.

Pipeline
1. Capture face using webcam and detect using Haar Cascade
2. Resize face and extract LBP histogram features
3. Store normalized histograms and names in known_faces.pkl
4. Perform live face detection and compute feature vectors
5. Match against stored features using Euclidean distance
6. Recognize users or label as unknown
7. Trigger unlock state when a valid match is found

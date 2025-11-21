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




Object Detection

This project performs automatic detection and classification of pens and pencils from an input image using classical computer vision. The system is orientation independent meaning that objects can appear rotated in any direction. The classification is done using a combination of contour geometry, aspect ratio analysis, solidity filtering, and dominant color extraction in HSV space. The output is an annotated image showing rotated bounding boxes and predicted labels for each detected object.

Processing Pipeline

1. Image Preprocessing
The image is resized if needed and converted to grayscale. Gaussian blurring and adaptive thresholding are applied to handle noise and uneven lighting. Morphological closing removes small gaps to create solid object regions.

2. Contour Extraction
External contours are detected from the thresholded mask. Small contours are filtered out to avoid noise. Each remaining contour is processed using a minimum area bounding rectangle which provides orientation, width, height, and angle.

3. Feature Extraction
The long and short sides of the rotated rectangle are used to compute the aspect ratio. The region inside the rotated rectangle is extracted and converted to HSV to compute the dominant color using k means clustering.

4. Object Classification
Classification is rule based using predefined aspect ratio ranges for pens and pencils. When the aspect ratio is ambiguous, the system falls back to color based heuristics using hue and saturation characteristics typical of wooden pencils and plastic pens.

5. Annotation and Output
Rotated bounding boxes are drawn on the original image along with predicted labels. A summary showing the count of pens pencils and unknown objects is added. The final annotated image is saved and can optionally be displayed.

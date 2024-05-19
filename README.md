# Face Recognition Project

## Overview
This project is designed for face detection and recognition using OpenCV, Haar cascades and SQL.

## Setup
1. Clone the repository.
2. Install the required dependencies using
    ```
    pip install -r requirements.txt
    ```
3. Ensure you have the necessary datasets and pre-trained models placed in the appropriate directories.

## Usage
- To train the model
    ```
    python scriptstrainer.py
    ```
- To run the face detection and recognition
    ```
    python scriptsdetector.py
    ```
- Alternatively, you can use the main script to select the mode
    ```
    python scriptsmain.py
    ```

## Files
- `main.py` Entry point to run the training or detection.
- `trainer.py` Script to train the face recognition model.
- `detector.py` Script to detect and recognize faces using the trained model.


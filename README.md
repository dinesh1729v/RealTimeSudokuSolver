# RealTimeSudokuSolver
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![TensorFlow/Keras](https://img.shields.io/badge/TF%2FKeras-CNN-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

Real-time Sudoku solver: detects the puzzle from a live camera feed, recognizes digits with a CNN, solves the board, and overlays the answers back on the video.

---

## ✨ What it does

- 🧠 **Digit recognition** via a CNN trained on **Chars74K**.
- 🧩 **Sudoku grid detection** using OpenCV geometry & contour tricks.
- 🔢 **9×9 cell extraction** and per-cell digit classification.
- 🧮 **Fast solving** using a classic algorithm (inspired by Peter Norvig).
- 🖊️ **AR overlay** of missing digits back onto the original frame.

---

## 🛠️ How It Works

1. **Create a model for digit recognition**  
   - Train a Convolutional Neural Network (CNN) using the **Chars74K dataset** for digits 0–9.  
   - Save the trained model as `digitRecognition.h5`.

2. **Detect the Sudoku grid**  
   - Use **OpenCV** to process frames in real time.  
   - Apply grayscale, blur, thresholding, and contour detection to isolate the Sudoku grid.  
   - Use perspective transformation to correct skew.

3. **Divide the grid into 9×9 cells**  
   - Split the warped grid image into 81 parts.  
   - Preprocess each cell and feed it into the CNN model to predict digits.  
   - Empty cells are marked as zeros.

4. **Solve the Sudoku puzzle**  
   - Represent the board as a 2D array and fill missing values using a solving algorithm.  
   - Uses **Peter Norvig’s Sudoku solver** .

5. **Overlay the solution**  
   - Using **OpenCV**, write the solved digits back onto the frame.  
   - Display the augmented frame in real time with the completed puzzle.

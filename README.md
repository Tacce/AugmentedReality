# Augmented Reality Planar Tracking
Project Work for the *Computer Vision and Image Processing* course at **University of Bologna**.

## Overview

This project utilizes **Python** and **OpenCV** to achieve geometric alignment and photometric consistency between real and virtual elements. The system implements two different tracking strategies and an adaptive color transfer algorithm to handle dynamic lighting conditions, overlaying a virtual layer onto a planar object (a book) within a video stream.

## Key Features

* **Feature-Based Tracking:** Utilizes **SIFT** for detection and **FLANN** for matching, refined by **RANSAC** homography estimation.
* **Dual Tracking Strategies:**
    * **Frame-to-Reference (F2R):** Matches current frame against the original reference image.
    * **Frame-to-Frame (F2F):** Incremental tracking concatenating homographies.
* **Performance Optimization:** Implements **ROI Tracking** to restrict search areas and reduce computational cost.
* **Photometric Consistency:** Features a statistical **Color Transfer** (Reinhard) in Lab space.
    * Includes **Soft Thresholding** to prevent color jitter in stationary conditions.
    * Adaptive luminance blending to handle contrast changes.

## Documentation

For a detailed explanation of the mathematical models, pipeline architecture, and full experimental analysis, please refer to the **[Project Report](Report.pdf)**.


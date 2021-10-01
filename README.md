# ECG-Numerical-Vs.-Image-Data-for-Emotion-Recognition-System

**Background**

The electrocardiogram (ECG) is a physiological signal used to diagnose and monitor cardiovascular disease, usually using ECG wave images. Numerous studies have proven that ECG can be used to detect human emotions using numerical data; however, ECG is typically captured as a wave image rather than as a numerical data. There is still no consensus on the effect of the ECG input format (either as an image or a numerical value) on the accuracy of the emotion recognition system (ERS). The ERS using ECG images is still inadequate. Therefore, this study compared ERS performance using ECG image and ECG numerical data to determine the effect of the ECG input format on the ERS.

**ECG Image**

The ECG Image.py file contains Python code for extracting image features and classifying the emotion. Features are extracted using Oriented FAST and Rotated BRIEF (ORB), Scale Invariant Feature Transform (SIFT), KAZE, Accelerated-KAZE (AKAZE), Binary Robust Invariant Scalable Keypoints (BRISK), and Histogram of Oriented Gradients (HOG).

**ECG Numerical**

The ECG Numerical.py file contains Python code used to classify emotion using features taken from the Augsburg BioSig Toolbox (J. Wagner, “Augsburg biosignal toolbox (aubt),” Univ. Augsbg., 2014).

**Python Version**

Python version 3.8.5 is used in this code.

**DOI**

https://doi.org/10.5281/zenodo.5542739

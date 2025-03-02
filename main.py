
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title
st.title("Fingerprint Comparison")

# Upload Fingerprint Images
uploaded_file1 = st.file_uploader("Upload First Fingerprint", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Upload Second Fingerprint", type=["png", "jpg", "jpeg"])

def match_fingerprints(img1, img2):
    # Convert PIL images to OpenCV format
    img1 = np.array(img1)
    img2 = np.array(img2)

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # ORB Detector
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score
    match_score = (len(matches) / max(len(keypoints1), len(keypoints2))) * 100
    return match_score, matches, keypoints1, keypoints2

if uploaded_file1 and uploaded_file2:
    img1 = Image.open(uploaded_file1).convert("RGB")
    img2 = Image.open(uploaded_file2).convert("RGB")

    # Perform fingerprint matching
    similarity, matches, keypoints1, keypoints2 = match_fingerprints(img1, img2)

    # Display images
    st.image([img1, img2], caption=["Fingerprint 1", "Fingerprint 2"], width=250)

    # Display similarity score
    st.success(f"Similarity Score: {similarity:.2f}%")

    # Draw matching keypoints
    img1_cv = np.array(img1)
    img2_cv = np.array(img2)
    img_matches = cv2.drawMatches(img1_cv, keypoints1, img2_cv, keypoints2, matches[:20], None, flags=2)

    # Show matched image
    st.image(img_matches, caption="Matched Keypoints", use_column_width=True)

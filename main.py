import cv2
import numpy as np
import streamlit as st
from PIL import Image

def compare_fingerprints(image1, image2):
    img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    similarity = (len(matches) / max(len(keypoints1), len(keypoints2))) * 100
    
    return similarity

st.set_page_config(page_title="Fingerprint Comparator", layout="centered")
st.title("🔍 Fingerprint Comparator")

uploaded_file1 = st.file_uploader("Upload First Fingerprint", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Upload Second Fingerprint", type=["png", "jpg", "jpeg"])

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Fingerprint 1", use_column_width=True)
    with col2:
        st.image(image2, caption="Fingerprint 2", use_column_width=True)
    
    if st.button("Compare Fingerprints"):
        similarity = compare_fingerprints(image1, image2)
        st.success(f"✅ Similarity: {similarity:.2f}%")


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
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image

# # Title
# st.title("Fingerprint Comparison")

# # Upload Fingerprint Images
# uploaded_file1 = st.file_uploader("Upload First Fingerprint", type=["png", "jpg", "jpeg"])
# uploaded_file2 = st.file_uploader("Upload Second Fingerprint", type=["png", "jpg", "jpeg"])

# def enhance_image(img):
#     """Enhances fingerprint contrast using CLAHE (Adaptive Histogram Equalization)."""
#     img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     return clahe.apply(img_gray)

# def match_fingerprints(img1, img2):
#     """Matches fingerprints using SIFT and calculates similarity score."""
#     # Enhance Images
#     img1_gray = enhance_image(img1)
#     img2_gray = enhance_image(img2)

#     # Use SIFT for feature detection
#     sift = cv2.SIFT_create()
#     keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

#     if descriptors1 is None or descriptors2 is None:
#         return 0, [], keypoints1, keypoints2, None  # No valid keypoints

#     # FLANN Matcher
#     index_params = dict(algorithm=1, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     # Apply ratio test
#     good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

#     # Compute similarity score based on distance of matches
#     if len(good_matches) > 0:
#         avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
#         similarity = max(0, 100 - avg_distance)  # Normalize similarity score
#     else:
#         similarity = 0

#     # Homography check (helps filter false matches)
#     if len(good_matches) > 10:
#         src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         if matrix is not None:
#             similarity *= 1.2  # Boost confidence if homography is valid

#     return min(100, similarity), good_matches, keypoints1, keypoints2, img1_gray, img2_gray

# if uploaded_file1 and uploaded_file2:
#     img1 = Image.open(uploaded_file1).convert("RGB")
#     img2 = Image.open(uploaded_file2).convert("RGB")

#     # Perform fingerprint matching
#     similarity, matches, keypoints1, keypoints2, img1_gray, img2_gray = match_fingerprints(img1, img2)

#     # Display images
#     st.image([img1, img2], caption=["Fingerprint 1", "Fingerprint 2"], width=250)

#     # Display similarity score
#     st.success(f"Similarity Score: {similarity:.2f}%")

#     # Draw matching keypoints if there are enough valid matches
#     if len(matches) > 0:
#         img_matches = cv2.drawMatches(img1_gray, keypoints1, img2_gray, keypoints2, matches[:20], None, flags=2)
#         st.image(img_matches, caption="Matched Keypoints", use_column_width=True)
#     else:
#         st.warning("No sufficient matches found. Try another fingerprint pair.")

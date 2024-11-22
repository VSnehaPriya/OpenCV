import cv2
import streamlit as st
import numpy as np
from PIL import Image as pil_img

# Load the face cascade
face_cascade = cv2.CascadeClassifier(r'C:\Users\E430384\Downloads\VS Code\CV\cascade_frontface_default.xml')

# Streamlit UI for image upload
st.title("Face Detection Web App")
st.write("Upload an image to detect faces.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = pil_img.open(uploaded_file)
    image = np.array(image)

    # Convert from RGB to BGR (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)

    # Draw rectangles around faces and count them
    face_count = len(faces)
    for x, y, w, h in faces:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)

    # Add text for face count
    cv2.putText(image, f'Faces Detected: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with detected faces
    st.image(image, channels="BGR", caption="Processed Image with Detected Faces")

else:
    st.write("Please upload an image.")

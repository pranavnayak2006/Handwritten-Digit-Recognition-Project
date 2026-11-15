
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #172337;
}

h1, h2, h3 {
    color: #60A5FA;
}

p, label {
    color: #E5E7EB;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = Path("models/digit_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    image.thumbnail((1000, 1000))

    img_array = np.array(image)

    border_pixels = np.concatenate([
        img_array[0, :],
        img_array[-1, :],
        img_array[:, 0],
        img_array[:, -1]
    ])

    # Make digit white and background black
    if np.mean(border_pixels) > 127:
        img_array = 255 - img_array

    threshold = max(40, int(np.max(img_array) * 0.25))

    binary = np.where(
        img_array > threshold,
        255,
        0
    ).astype(np.uint8)

    digit_image = Image.fromarray(binary)

    bbox = digit_image.getbbox()

    if bbox is None:
        return None, None

    digit_image = digit_image.crop(bbox)

    # Resize digit to fit inside 20x20
    digit_image.thumbnail((20, 20))

    # Create 28x28 black background
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2

    canvas.paste(digit_image, (x, y))

    canvas = canvas.filter(
        ImageFilter.GaussianBlur(radius=0.4)
    )

    processed = np.array(canvas).astype("float32") / 255.0

    processed = processed.reshape(1, 28, 28, 1)

    return processed, canvas


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_digit(image, source):

    processed, display_image = preprocess_image(image)

    if processed is None:
        st.error("No digit detected. Please try another image.")
        return

    prediction = model.predict(
        processed,
        verbose=0
    )

    predicted_digit = int(np.argmax(prediction[0]))

    confidence = float(np.max(prediction[0]) * 100)

    st.subheader("Processed Image")

    st.image(
        display_image.resize((280, 280)),
        width=280
    )

    st.success(
        f"Recognized Digit: {predicted_digit}"
    )

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # Store prediction history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Digit": predicted_digit,
        "Confidence": round(confidence, 2)
    })

    # Show probabilities
    st.subheader("Prediction Probabilities")

    probability_data = pd.DataFrame({
        "Digit": list(range(10)),
        "Probability": prediction[0] * 100
    })

    st.bar_chart(
        probability_data.set_index("Digit")
    )


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🔢 Digit AI")

st.sidebar.write(
    "Handwritten Digit Recognition System"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Draw Digit",
        "Upload Image",
        "Prediction History",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Deep Learning Mini Project"
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.title("🔢 Handwritten Digit Recognition")

    st.subheader("Welcome to Digit AI!")

    st.write("""
    This application uses a Convolutional Neural Network (CNN)
    to recognize handwritten digits from 0 to 9.

    You can draw a digit or upload an image.
    The trained AI model will predict the digit.
    """)

    st.info(
        "Choose Draw Digit or Upload Image from the sidebar to begin."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Digit Classes", "10")
    col2.metric("Dataset", "MNIST")
    col3.metric("Model", "CNN")

    st.subheader("Main Features")

    c1, c2 = st.columns(2)

    with c1:
        st.write("✏️ Draw handwritten digits")
        st.write("📤 Upload digit images")
        st.write("📊 View prediction probabilities")

    with c2:
        st.write("📜 Prediction history")
        st.write("📈 Accuracy and loss graphs")
        st.write("🤖 Deep learning recognition")


# ==========================================
# DRAW DIGIT PAGE
# ==========================================

elif page == "Draw Digit":

    st.title("✏️ Draw Your Digit")

    st.write(
        "Draw a single digit from 0 to 9 in the black box."
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    if st.button("Recognize Drawn Digit"):

        if canvas_result.image_data is not None:

            drawing = canvas_result.image_data.astype(np.uint8)

            drawing_image = Image.fromarray(
                drawing
            ).convert("RGB")

            predict_digit(
                drawing_image,
                "Drawing"
            )

        else:
            st.warning("Please draw a digit first.")


# ==========================================
# UPLOAD IMAGE PAGE
# ==========================================

elif page == "Upload Image":

    st.title("📤 Upload Handwritten Digit")

    st.write(
        "Upload an image containing one handwritten digit."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            width=250
        )

        if st.button("Recognize Uploaded Digit"):

            predict_digit(
                image,
                "Image Upload"
            )


# ==========================================
# PREDICTION HISTORY PAGE
# ==========================================

elif page == "Prediction History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Predictions",
            len(df)
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No predictions yet. Draw or upload a digit first."
        )


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Training performance of the CNN model."
    )

    accuracy_path = Path("accuracy.png")
    loss_path = Path("loss.png")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Accuracy Graph")

        if accuracy_path.exists():

            st.image(
                str(accuracy_path),
                use_container_width=True
            )

        else:

            st.warning("accuracy.png was not found.")

    with col2:

        st.subheader("Loss Graph")

        if loss_path.exists():

            st.image(
                str(loss_path),
                use_container_width=True
            )

        else:

            st.warning("loss.png was not found.")

    st.subheader("Model Information")

    st.write("Architecture: CNN")
    st.write("Dataset: MNIST")
    st.write("Input Image Size: 28 × 28")
    st.write("Output Classes: 0–9")


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("📘 About the Project")

    st.subheader(
        "Handwritten Digit Recognition Using Deep Learning"
    )

    st.write("""
    This project recognizes handwritten numerical digits
    using a Convolutional Neural Network.

    The MNIST dataset is used to train the model.
    The trained model predicts digits from uploaded images
    and drawings.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - TensorFlow and Keras
    - Streamlit
    - NumPy
    - Pillow
    - Pandas
    - Streamlit Drawable Canvas
    - MNIST Dataset
    """)

    st.subheader("Working Procedure")

    st.write("""
    1. User draws or uploads a handwritten digit.
    2. The image is converted into grayscale.
    3. Image preprocessing converts it into 28 × 28 pixels.
    4. CNN model processes the image.
    5. The system predicts the digit.
    6. Prediction and confidence are displayed.
    """)

    st.subheader("Student Details")

    st.write("**Student Name:** Pranav Nayak")

    st.write(
        "**Department:** Robotics and Artificial Intelligence"
    )

    st.write(
        "**College:** Mangalore Institute of Technology and Engineering (MITE)"
    )

    st.write(
        "**Project Type:** College Mini Project"
    )

    st.divider()

    st.caption(
        "Handwritten Digit Recognition | Deep Learning Mini Project"
    )

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal  # Assuming these are markdown strings or functions
import tempfile

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS for a modern and good-looking design
st.markdown("""
    <style>
    /* General Layout */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #f5f7fb;
        color: #333;
    }

    /* Navbar Styling */
    .navbar {
        background: linear-gradient(90deg, #2c3e50, #34495e);  /* Gradient dark blue */
        padding: 15px 20px;
        margin-bottom: 30px;
        border-radius: 0 0 8px 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .navbar button {
        margin-right: 15px;
        padding: 12px 25px;
        background-color: #3498db;  /* Vibrant blue */
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    .navbar button:hover {
        background-color: #2980b9;  /* Darker blue on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .navbar button:active {
        background-color: #1f6391;  /* Even darker when clicked */
        transform: translateY(0);
    }

    /* Content Container */
    .content-container {
        padding: 25px 35px;
        max-width: 1000px;
        margin: 0 auto;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .content-container:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    /* Headings */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 20px;
    }
    h4 {
        color: #34495e;
        font-weight: 600;
    }

    /* Image Styling */
    .stImage {
        max-width: 85%;
        margin: 20px auto;
        display: block;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* File Uploader */
    .stFileUploader {
        width: 100%;
        padding: 20px;
        margin: 20px 0;
        border: 2px dashed #95a5a6;
        border-radius: 10px;
        background-color: #fafbfc;
        transition: border-color 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #3498db;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1abc9c;  /* Teal */
        color: white;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #16a085;  /* Darker teal */
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Expander */
    .stExpander {
        margin-top: 20px;
    }
    .stExpander>div {
        padding: 20px;
        background-color: #f9fbfc;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Success Message */
    .stSuccess {
        background-color: #dff0d8;
        color: #3c763d;
        border-radius: 6px;
        padding: 10px;
        font-weight: 500;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .navbar {
            padding: 10px;
        }
        .navbar button {
            margin: 5px 0;
            padding: 10px 15px;
            width: 100%;
        }
        .content-container {
            padding: 15px;
            max-width: 100%;
        }
        .stImage {
            max-width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Navbar
cols = st.columns(3)
pages = ["Home", "About", "Disease Identification"]
selected_page = None

with st.container():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    for i, page in enumerate(pages):
        with cols[i]:
            if st.button(page, key=page, help=f"Go to {page}", use_container_width=True):
                selected_page = page
    st.markdown('</div>', unsafe_allow_html=True)

# Use session state to persist the selected page
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"  # Default page

if selected_page:
    st.session_state.app_mode = selected_page

app_mode = st.session_state.app_mode

# Home Page
if app_mode == "Home":
    st.title("OCT Retinal Analysis Platform")
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.markdown("""
        #### Welcome to the Retinal OCT Analysis Platform

        **Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, enabling early detection and monitoring of retinal diseases. Over 30 million OCT scans are performed annually, aiding in the diagnosis and management of conditions like choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

        ##### Why OCT Matters
        OCT is a non-invasive tool critical to ophthalmology, helping detect retinal abnormalities efficiently. This platform leverages advanced machine learning to streamline OCT analysis, reducing the burden on medical professionals and enhancing diagnostic accuracy.

        ---
        #### Key Features
        - **Automated Image Analysis**: Classifies OCT images into **Normal**, **CNV**, **DME**, and **Drusen** using cutting-edge models.
        - **Cross-Sectional Imaging**: View high-quality retinal scans for clinical decision-making.
        - **Streamlined Workflow**: Upload, analyze, and review scans in just a few steps.
        ---
        #### Get Started
        - Navigate to **Disease Identification** to upload and analyze OCT scans.
        - Visit **About** to learn more about the dataset and methodology.
        - Contact us at [support@example.com](#) for assistance or inquiries.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.title("About the Retinal OCT Analysis Platform")
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.markdown("""
        #### About the Dataset
        Retinal Optical Coherence Tomography (OCT) captures detailed cross-sectional images of the retina. With approximately 30 million scans performed yearly, analyzing these images is time-intensive. Our platform automates this process using a dataset of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets across four categories:
        - **Normal**
        - **Choroidal Neovascularization (CNV)**: Neovascular membrane with subretinal fluid.
        - **Diabetic Macular Edema (DME)**: Retinal thickening with intraretinal fluid.
        - **Drusen (Early AMD)**: Multiple drusen deposits.

        Images are labeled as `(disease)-(randomized patient ID)-(image number)` and sourced from leading institutions like the Shiley Eye Institute (UC San Diego), California Retinal Research Foundation, and others between July 2013 and March 2017.

        ---
        #### Data Validation Process
        Each image underwent a rigorous tiered grading system:
        1. **First Tier**: Trained students excluded low-quality images (e.g., severe artifacts).
        2. **Second Tier**: Four ophthalmologists independently labeled images for CNV, DME, drusen, and other pathologies.
        3. **Third Tier**: Two senior retinal specialists (20+ years of experience) verified labels.

        A validation subset of 993 scans was separately graded by two ophthalmologists, with discrepancies resolved by a senior specialist, ensuring high accuracy.

        ---
        #### Purpose
        This platform aims to assist clinicians by providing reliable, automated OCT analysis, drawing from a diverse, expertly validated dataset.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Disease Identification (Prediction) Page
elif app_mode == "Disease Identification":
    st.title("Retinal OCT Disease Identification")
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        st.markdown("Upload an OCT scan to classify it as **Normal**, **CNV**, **DME**, or **Drusen**.")

        # File uploader
        test_image = st.file_uploader("Upload your OCT Image:", type=["jpg", "jpeg", "png"])
        if test_image is not None:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
                tmp_file.write(test_image.read())
                temp_file_path = tmp_file.name

            # Display uploaded image
            st.image(test_image, caption="Uploaded OCT Scan", use_column_width=True)

            # Predict button
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    result_index = model_prediction(temp_file_path)
                    class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    st.success(f"Prediction: **{class_name[result_index]}**")

                    # Detailed explanation in expander
                    with st.expander("Learn More About the Prediction"):
                        if result_index == 0:  # CNV
                            st.write("OCT scan showing *CNV with subretinal fluid*.")
                            st.markdown(cnv)
                        elif result_index == 1:  # DME
                            st.write("OCT scan showing *DME with retinal thickening and intraretinal fluid*.")
                            st.markdown(dme)
                        elif result_index == 2:  # DRUSEN
                            st.write("OCT scan showing *drusen deposits in early AMD*.")
                            st.markdown(drusen)
                        elif result_index == 3:  # NORMAL
                            st.write("OCT scan showing a *normal retina with preserved foveal contour*.")
                            st.markdown(normal)
        st.markdown('</div>', unsafe_allow_html=True)

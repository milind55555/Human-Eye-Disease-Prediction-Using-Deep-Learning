import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import bcrypt
import pymongo
import tempfile
from recommendation import cnv, dme, drusen, normal

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://hanchatemilind291:mJtJKOX3XB4wTXFa@cluster0.dbvnnt5.mongodb.net/")  # Updated DB connection
users_db = client["eye_disease_db"]
users_collection = users_db["users"]

# Password Hashing Function
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Password Verification Function
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Store User in Database
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False  # User already exists
    hashed_pw = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed_pw})
    return True

# Authenticate User
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user["password"]):
        return True
    return False

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)

# Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.title("Login to OCT Analysis Platform")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")

    st.subheader("New User? Register Below")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(new_username, new_password):
            st.success("Registration successful! Please log in.")
        else:
            st.error("Username already exists.")

def main_page():
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    page = st.sidebar.radio("Navigation", ["Home", "About", "Disease Prediction"])
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    if page == "Home":
        st.title("OCT Retinal Analysis Platform")
        
        st.header("Welcome to the Retinal OCT Analysis Platform")
        st.write("""
        **Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).
        
        #### Why OCT Matters
        OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.
        """)
        
        st.subheader("Key Features of the Platform")
        st.write("""
        - **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
        - **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
        - **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.
        """)
        
    elif page == "About":
        st.title("About the Dataset")
        st.write("""
        Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
        Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
        """)
        
        st.subheader("Content")
        st.write("""
        The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL, CNV, DME, DRUSEN). 
        There are 84,495 OCT images (JPEG) and 4 categories (NORMAL, CNV, DME, DRUSEN).
        
        Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.
        """)
        
    elif page == "Disease Prediction":
        st.title("Disease Prediction")
        test_image = st.file_uploader("Upload your OCT Image:", type=["jpg", "jpeg", "png"])
        if test_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
                tmp_file.write(test_image.read())
                temp_file_path = tmp_file.name
            st.image(test_image, caption="Uploaded OCT Scan", use_column_width=True)
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    result_index = model_prediction(temp_file_path)
                    class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    st.success(f"Prediction: **{class_name[result_index]}**")
                    with st.expander("Learn More About the Prediction"):
                        st.markdown([cnv, dme, drusen, normal][result_index])

# App Navigation
if st.session_state.logged_in:
    main_page()
else:
    login_page()

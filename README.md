# Human-Eye-Disease-Prediction-Using-Deep-Learning
Overview
This project implements deep learning-based eye disease prediction using MobileNetV3-Large, a lightweight yet powerful convolutional neural network (CNN). The model is optimized for mobile and edge devices, making it efficient for real-time medical diagnosis.

The system can classify diabetic retinopathy, glaucoma, age-related macular degeneration (AMD), and normal eyes using retinal images. It aims to assist ophthalmologists by providing automated and accurate disease detection.

Why MobileNetV3-Large?
MobileNetV3 is a highly optimized CNN architecture designed for low-latency and high-accuracy applications, making it ideal for medical imaging tasks.

Lightweight: Faster inference on mobile and embedded devices.

Efficient: Uses depthwise separable convolutions to reduce computational cost.

High Accuracy: Optimized with SE blocks (Squeeze-and-Excitation) and hard-swish activation for improved feature learning.

Better Performance: Outperforms older models like VGG16 in terms of speed while maintaining competitive accuracy.

Features
✅ Deep Learning-based Eye Disease Detection
✅ MobileNetV3-Large for fast and efficient classification
✅ Pretrained on large retinal image datasets
✅ Supports multiple eye diseases (Diabetic Retinopathy, Glaucoma, AMD)
✅ Optimized for deployment on mobile and cloud platforms
✅ User-friendly Web UI using Streamlit

Dataset
The model is trained on publicly available retinal image datasets including:

APTOS 2019 Blindness Detection

EyePACS Dataset

OCT and Fundus Imaging Datasets

Images are categorized into four classes:
1️⃣ Normal – No disease present
2️⃣ Diabetic Retinopathy – Retina damage due to diabetes
3️⃣ Glaucoma – Optic nerve damage from increased eye pressure
4️⃣ AMD (Age-Related Macular Degeneration) – Deterioration of the central retina

Technologies Used
Python 🐍

TensorFlow/Keras (for deep learning)

MobileNetV3-Large (for image classification)

OpenCV (for image preprocessing)

Streamlit (for web-based deployment)

Model Training Process
1️⃣ Data Preprocessing
✔ Image resizing to 224x224 (as required by MobileNetV3)
✔ Normalization and augmentation (rotation, flipping, contrast adjustment)

2️⃣ Model Architecture - MobileNetV3-Large
✔ Pretrained weights (transfer learning)
✔ Modified last layers for classification
✔ Softmax activation for multi-class output

3️⃣ Training Strategy
✔ Optimizer: Adam
✔ Loss Function: Categorical Crossentropy
✔ Batch Size: 32
✔ Epochs: 15

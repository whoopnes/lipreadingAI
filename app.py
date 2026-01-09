import streamlit as st
import mediapipe as mp
import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import gdown
import zipfile

LIP_WIDTH = 112
LIP_HEIGHT = 80
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
DATASET = 'Dataset_Indo'
DATASET_ZIP = "Dataset_Indo.zip"
DATASET_URL = "https://drive.google.com/drive/folders/1gVvOMczguUKT57p0oXIq9LYP_Ff82L6n?usp=sharing"
LABEL_DICT = {0: 'air', 1: 'bakso', 2: 'doa', 3: 'kopi', 4: 'novel', 5: 'puding', 6: 'rumus', 7: 'surat'}
img_path = 'background.webp'

def set_bg(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp{{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.5);  
            pointer-events: none;
            z-index: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg(img_path)

def ensure_dataset():
    if not os.path.exists(DATASET):
        st.info("Downloading Dataset Indo...")
        gdown.download(DATASET_URL, DATASET_ZIP, quiet=False)
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(DATASET_ZIP)

@st.cache_resource
def load_model(model_type):
    if model_type == "Augmented":
        model_path = "model_3DCNN+LSTM with Aug.keras"
    else:
        model_path = "model_3DCNN+LSTM.keras"

    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model loaded: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def avail_videos(DATASET):
    videos_dict = {}
    if not os.path.exists(DATASET):
        return videos_dict
    
    for label in os.listdir(DATASET):
        label_path = os.path.join(DATASET, label)
        if not os.path.isdir(label_path):
            continue
        videos_dict[label] = []

        for sample in os.listdir(label_path):
            sample_path = os.path.join(label_path, sample)
            if not os.path.isdir(sample_path):
                continue
            
            video_path = os.path.join(sample_path, "video.mp4")
            if os.path.exists(video_path):
                videos_dict[label].append({"sample": sample, "path": video_path})
    return videos_dict

def crop_data_lip(frame, face_mesh):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0].landmark

    mouth_pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in MOUTH_LANDMARKS
    ])

    x_min = np.min(mouth_pts[:, 0])
    x_max = np.max(mouth_pts[:, 0])
    y_min = np.min(mouth_pts[:, 1])
    y_max = np.max(mouth_pts[:, 1])

    margin_w = int((x_max - x_min) * 0.5)  
    margin_h = int((y_max - y_min) * 0.6)  

    x_min = max(0, x_min - margin_w)
    x_max = min(w, x_max + margin_w)
    y_min = max(0, y_min - margin_h)
    y_max = min(h, y_max + margin_h)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    target_ratio = LIP_WIDTH / LIP_HEIGHT
    current_ratio = crop_w / crop_h

    if current_ratio > target_ratio:
        new_h = int(crop_w / target_ratio)
        diff = new_h - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    else:
        new_w = int(crop_h * target_ratio)
        diff = new_w - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    lip_frame = frame[y_min:y_max, x_min:x_max]
    return lip_frame if lip_frame.size > 0 else None

def preprocess_data(video_path, progress_placeholder):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_frame = []
    mp_face = mp.solutions.face_mesh

    with mp_face.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        refine_landmarks = True
    ) as face_mesh:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress = int((frame_idx / frame_count) * 100)
            progress_placeholder.progress(progress, text=f"Sedang memproses frame {frame_idx + 1}/{frame_count}")

            lip_frame = crop_data_lip(frame, face_mesh)
            if lip_frame is not None:
                lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

                lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)

                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
                l_channel_eq = clahe.apply(l_channel)

                lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
                lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)

                lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
                lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)

                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]
                ])

                lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
                lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
                process_frame.append(lip_frame_eq)
            frame_idx += 1
    cap.release()
    return np.array(process_frame)

def predicting_word(model, frames):
    try:
        frames = frames.astype('float32') / 255.0
        input_data = np.expand_dims(frames, axis=0)
        prediction = model.predict(input_data, verbose=0)
        
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        predicted_word = LABEL_DICT[predicted_class]
        
        return predicted_word, confidence, predicted_class
    except Exception as e:
        print(f"Error: {e}")
        return "Error", 0.0, -1

st.title("👄Lip Reading AI")
st.subheader("Testing Lip Reading Model Using Video🎥")
st.divider()
ensure_dataset()

videos_dict = avail_videos(DATASET)
if not videos_dict:
    st.error("Can't found the video")
    st.stop()

model_type = st.selectbox("Model", ["Non-Augmented", "Augmented"])
avail_labels = sorted([label for label, samples in videos_dict.items() if len(samples)])
selected_label = st.selectbox("Label", avail_labels)

samples = videos_dict[selected_label]
sample_names = [s["sample"] for s in samples]
selected_sample = st.selectbox("Video Sample", sample_names)

video_data = next(s for s in samples if s["sample"] == selected_sample)
video_path = video_data["path"]

st.video(video_path)
st.subheader(f"Ground Truth: **{selected_label}**")
st.divider()

if st.button("Process & Prediction", use_container_width=True):
    with st.spinner("Processing the video..."):
        model = load_model(model_type)
        progress = st.empty()
        frames = preprocess_data(video_path, progress)
        progress.empty()

    if len(frames) == 0:
        st.error("There is no face datected")
        st.stop()

    predicted_word, confidence, predicted_class = predicting_word(model, frames)

    st.divider()
    st.subheader("Result")

    st.write(f"-- Prediction: {predicted_word} --")
    st.write(f"-- Confidence: {confidence:.2%} --")

    if predicted_word == selected_label:
        st.success("Correct Prediction")
    else:
        st.error("Wrong Prediction")

    st.progress(confidence)
    st.subheader("Frame Example")

    cols = st.columns(3)
    num_frames = min(6, len(frames))
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    for i, idx in enumerate(indices):
        with cols[i % 3]:
            img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            st.image(img, use_container_width=True)

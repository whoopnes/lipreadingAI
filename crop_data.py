import cv2
import mediapipe as mp
import numpy as np
import os
import imageio.v2 as imageio

DATASET = 'Dataset_Indo'
TARGET_DIR = 'Dataset Dikit'

LIP_WIDTH = 112
LIP_HEIGHT = 80

os.makedirs(TARGET_DIR, exist_ok=True)

MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

mp_face = mp.solutions.face_mesh
counts = {}

for label in os.listdir(DATASET):
    label_path = os.path.join(DATASET, label)
    if not os.path.isdir(label_path):
        continue
    for sample in os.listdir(label_path):
        sample_path = os.path.join(label_path, sample)
        if not os.path.isdir(sample_path):
            continue
        
        video_path = os.path.join(sample_path, "video.mp4")
        if not os.path.exists(video_path):
            print("Video not found:", video_path)
            continue

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_dir = os.path.join(TARGET_DIR, label, sample)
        os.makedirs(output_dir, exist_ok=True)
        frames_video = []

        with mp_face.FaceMesh(static_image_mode = False, max_num_faces = 1, refine_landmarks = True) as face_mesh:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
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

                    if lip_frame.size == 0:
                        continue

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
                    lip_frame = lip_frame_eq

                    cv2.imwrite(os.path.join(output_dir, f"{frame_idx:03d}.png"), lip_frame)

                    frames_video.append(lip_frame)
                    frame_idx += 1
        cap.release()
        video_out = os.path.join(output_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            video_out,
            fourcc,
            fps,
            (LIP_WIDTH, LIP_HEIGHT)
        )

        for f in frames_video:
            out.write(f)
        out.release()

        data_array = []
        for f in frames_video:
            frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pixels = frame.reshape(-1, 3).tolist()
            data_array.append(pixels)

        data_txt = os.path.join(output_dir, "data.txt")
        with open(data_txt, 'w') as txt_file:
            txt_file.write(str(data_array))
        

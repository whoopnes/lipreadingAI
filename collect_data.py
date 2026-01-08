import os
import shutil

DATASET = "Dataset Lip Reading"
TARGET_DIR = "Dataset_Indo"

os.makedirs(TARGET_DIR, exist_ok=True)

counts = {}

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)
    if not os.path.exists(person_path):
        continue

    for video in os.listdir(person_path):
        if not video.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        
        label = os.path.splitext(video)[0]
        label = label.split('_')[0]

        counts.setdefault(label, 0)
        counts[label] += 1

        label_dir = os.path.join(TARGET_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        video_folder = f"{label}_{counts[label]:03d}"
        video_dir = os.path.join(label_dir, video_folder)
        os.makedirs(video_dir, exist_ok=True)

        src = os.path.join(person_path, video)
        dst = os.path.join(video_dir, "video.mp4")

        shutil.move(src, dst)




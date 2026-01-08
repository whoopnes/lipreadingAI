# 👄 LipReading AI: Visual Speech Recognition System

LipReading AI adalah proyek Deep Learning yang bertujuan untuk mengenali ucapan manusia hanya dari gerakan bibir tanpa menggunakan input suara. Sistem ini menggunakan kombinasi 3D Convolutional Neural Network (3D CNN) dan Bidirectional LSTM untuk mengekstraksi informasi visual dan temporal dari video.

Model ini mampu mengklasifikasikan gerakan bibir ke dalam 8 kelas kata:

🧊 air
🍜 bakso
🙏 doa
☕ kopi
📖 novel
🍮 puding
🧮 rumus
✉️ surat

# 📊 Dataset
Proyek ini menggunakan dataset open-access dari Kaggle:
Indonesia Lip Reading Dataset
https://www.kaggle.com/datasets/luthfiaca/indonesia-lipreading-468landmarksmediapipe-dataset

Dataset berisi kumpulan video beberapa orang yang mengucapkan kata tertentu. Dataset melibatkan lebih dari satu subjek sehingga mencakup variasi bentuk wajah, gaya bicara, dan gerakan bibir.

# 🧠 Model Architecture
3D CNN → Ekstraksi fitur spasial & temporal dari video
LSTM → Pemodelan urutan gerakan bibir
Classifier → Fully Connected + Softmax (8 kelas)

# 📈 Performance
Metric	Value
Test Accuracy	98.93%
Macro F1-Score	0.99
Weighted F1-Score	0.99
Model menunjukkan performa tinggi pada dataset uji, namun masih memiliki keterbatasan dalam generalisasi terhadap wajah di luar dataset.

# 🚀 Installation & Usage
## 1️⃣ Clone Repository
git clone https://github.com/yourusername/LipReadingAI.git
cd LipReadingAI
## 2️⃣ Install Dependencies
pip install -r requirements.txt
## 3️⃣ Run the Web App
streamlit run app.py

## ⚠️ Sistem saat ini belum mendukung real-time dan hanya menerima unggahan video yang mengikuti format dataset.

# 🧩 Limitations & Future Work
- Belum mendukung real-time inference
- Generalisasi terhadap wajah di luar dataset masih terbatas
- Membutuhkan dataset lebih besar & beragam
- Optimasi model agar lebih ringan untuk deployment

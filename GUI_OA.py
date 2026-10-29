import tensorflow as tf
import numpy as np
import os
import time
import cv2
from PIL import Image
from flask import Flask, request, render_template, make_response
from tensorflow.keras.losses import Loss

# Definisi Layer CBAM
class CBAM(tf.keras.layers.Layer):
    def __init__(self, filters, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            "filters": self.filters,
            "reduction_ratio": self.reduction_ratio,
        })
        return config

    def build(self, input_shape):
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.shared_dense_one = tf.keras.layers.Dense(self.filters // self.reduction_ratio, activation='relu')
        self.shared_dense_two = tf.keras.layers.Dense(self.filters)
        self.conv_spatial = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, input_feature):
        avg_pool = self.avg_pool(input_feature)
        max_pool = self.max_pool(input_feature)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
        channel_refined = input_feature * channel_attention

        avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.conv_spatial(tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1))

        return channel_refined * spatial_attention

# Inisialisasi Flask
app = Flask(__name__)

# Load Model
MODEL_PATH = r"C:\Users\Asus\PENELITIAN_RINI_VAMULIA_OA_CNN\Flask_App\bestcbam3_v2.h5"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'CBAM': CBAM})

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SAVE_IMAGE_FOLDER = 'static/saved_images'
os.makedirs(SAVE_IMAGE_FOLDER, exist_ok=True)

CSV_FILE = 'data_klasifikasi.csv'

class_names = ['Grade0', 'Grade1', 'Grade2', 'Grade3', 'Grade4']

# Fungsi Preprocessing
def preprocess_image(image_bytes_or_path):
    if isinstance(image_bytes_or_path, bytes):
        nparr = np.frombuffer(image_bytes_or_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img_pil = Image.open(image_bytes_or_path).convert('RGB')
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (224, 224))

    # CLAHE preprocessing
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img_clahe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalisasi [-1, 1]
    img_clahe = img_clahe.astype(np.float32)
    img_clahe = (img_clahe / 127.5) - 1.0

    img_clahe = np.expand_dims(img_clahe, axis=0)
    return img_clahe

# ROUTING

@app.route("/")
def home():
    return render_template("gui1.html")

@app.route("/777777")
def gui1():
    return render_template("gui1.html")

@app.route('/gui2')
def gui2():
    grade = -1
    probabilities = [0, 0, 0, 0, 0]
    return render_template('gui2.html', grade=grade, probabilities=probabilities)

@app.route('/predict-gui2', methods=['POST'])
def predict_gui2():
    print("Received request for prediction")  # Log untuk debugging
    if 'image' not in request.files:
        print("No image found in request")  # Log jika tidak ada gambar
        return {"grade": None, "error": "Gambar tidak ditemukan."}, 400
    file = request.files['image']
    nama_pasien = request.form.get('nama')
    tanggal_pemeriksaan = request.form.get('tgl_pemeriksaan')
    nama_dokter = request.form.get('dokter')

    try:
        import re
        nama_bersih = re.sub(r'[^a-zA-Z0-9]', '_', nama_pasien.upper())

        filename = f"{nama_bersih}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        if not os.path.exists(filepath):
            return {"grade": None, "error": "File gagal disimpan."}, 400

        input_img = preprocess_image(filepath)

        # Cek isi gambar yang dikirim
        print("Citra shape:", input_img.shape)
        print("Min:", np.min(input_img), "Max:", np.max(input_img))

        # Prediksi
        preds = model.predict(input_img)
        print("Output prediksi (softmax):", preds[0])

        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        probabilities = preds[0].tolist()


        return {
            "grade": idx,
            "class_name": class_names[idx],
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
            "filename": filename,
            "error": None
        }
    except Exception as e:
        return {"grade": None, "error": f"Kesalahan saat memproses gambar: {str(e)}"}, 500


@app.route('/save', methods=['POST'])
def save_data():
    import shutil

    nama_pasien = request.form.get('nama')
    tanggal_pemeriksaan = request.form.get('tgl_pemeriksaan')
    nama_dokter = request.form.get('dokter')
    hasil_klasifikasi = request.form.get('hasil_klasifikasi')
    filename = request.form.get('filename')

    if not all([nama_pasien, tanggal_pemeriksaan, nama_dokter, hasil_klasifikasi, filename]):
        return make_response("Data tidak lengkap", 400)

    # Lokasi gambar asli (hasil upload awal)
    src = os.path.join(UPLOAD_FOLDER, filename)

    # Lokasi folder penyimpanan final
    dst = os.path.join(SAVE_IMAGE_FOLDER, filename)

    print("COPY GAMBAR DARI:", src)
    print("KE:", dst)
    print("EXISTS?", os.path.exists(src))

    try:
        shutil.copy(src, dst)
    except Exception as e:
        return make_response(f"Gagal menyimpan gambar: {str(e)}", 500)

    # SIMPAN DATA KE CSV
    with open('hasil_klasifikasi.csv', 'a', newline='') as f:
        f.write(f"{nama_pasien},{tanggal_pemeriksaan},{nama_dokter},{hasil_klasifikasi},{filename}\n")

    return make_response("Data + gambar berhasil disimpan", 200)

@app.errorhandler(404)
def not_found(error):
    return "Halaman tidak ditemukan (404)", 404

if __name__ == "__main__":
    app.run(debug=True, port=7777)
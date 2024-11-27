from flask import Flask, jsonify, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model  # type: ignore

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model, scaler, dan label encoder
model = load_model('model/model_status_gizi.h5')  # Memuat model TensorFlow
scaler = joblib.load('model/scaler_status_gizi.pkl')  # Memuat scaler
label_encoder = joblib.load('model/label_encoder_status_gizi.pkl')  # Memuat label encoder

# Endpoint untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')  # Render file index.html dari folder templates

# Endpoint untuk prediksi status gizi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari request JSON
        data = request.json
        
        # Validasi data input
        if not data or 'Umur' not in data or 'Jenis Kelamin' not in data or 'Tinggi Badan' not in data:
            return jsonify({'error': 'Missing required fields: Umur, Jenis Kelamin, and Tinggi Badan'}), 400
        
        # Mengambil parameter input
        umur = data['Umur']
        jenis_kelamin = data['Jenis Kelamin']
        tinggi_badan = data['Tinggi Badan']
        
        # Mapping jenis kelamin ke angka (0: perempuan, 1: laki-laki)
        jenis_kelamin = 0 if jenis_kelamin == 'perempuan' else 1
        
        # Memproses input data
        input_data = np.array([[umur, jenis_kelamin, tinggi_badan]])
        input_data_scaled = scaler.transform(input_data)  # Normalisasi
        
        # Prediksi status gizi
        predictions = model.predict(input_data_scaled)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Mengembalikan hasil prediksi sebagai JSON
        return jsonify({
            'Umur': umur,
            'Jenis Kelamin': 'Laki-Laki' if jenis_kelamin == 1 else 'Perempuan',
            'Tinggi Badan': tinggi_badan,
            'Prediksi Status Gizi': predicted_label
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Menjalankan aplikasi Flask
    app.run(debug=True)

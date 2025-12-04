import os
import tensorflow as tf
import numpy as np

# Lokasi output model dari pipeline
SERVING_MODEL_DIR = 'output/serving_model'

def find_latest_model_dir(base_dir):
    """Mencari folder versi model paling baru (angka terbesar)."""
    if not os.path.exists(base_dir):
        raise Exception(f"Folder {base_dir} tidak ditemukan. Jalankan pipeline dulu!")
        
    versions = [int(v) for v in os.listdir(base_dir) if v.isdigit()]
    if not versions:
        raise Exception("Tidak ada model tersimpan di folder serving_model.")
        
    latest_version = max(versions)
    return os.path.join(base_dir, str(latest_version))

def predict():
    # 1. Load Model
    model_path = find_latest_model_dir(SERVING_MODEL_DIR)
    print(f"=== Memuat Model dari: {model_path} ===")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model Berhasil Dimuat!")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return

    # 2. Siapkan Data Dummy (Simulasi data yang sudah di-transform)
    # Ingat: Model ini mengharapkan input yang sudah berupa Z-Score (angka desimal kecil)
    # dan Age_bucket berupa integer.
    
    print("\n=== Melakukan Prediksi dengan Data Dummy ===")
    
    # Kita buat input dictionary
    input_data = {
        'Pregnancies': np.array([[0.5]]),           # Contoh z-score
        'Glucose': np.array([[1.2]]),               # Gula darah tinggi (diatas rata-rata)
        'BloodPressure': np.array([[-0.2]]),        # Tekanan darah normal
        'SkinThickness': np.array([[0.1]]),
        'Insulin': np.array([[-0.5]]),
        'BMI': np.array([[0.8]]),                   # BMI agak tinggi
        'DiabetesPedigreeFunction': np.array([[0.3]]),
        'Age': np.array([[1.5]]),                   # Umur tua (z-score positif)
        'Age_bucket': np.array([[2]], dtype=int)    # Kategori Lansia (misal bucket 2)
    }

    # 3. Prediksi
    prediction = model.predict(input_data)
    
    # 4. Tampilkan Hasil
    probability = prediction[0][0]
    print(f"\nSkor Probabilitas Diabetes: {probability:.4f}")
    
    if probability > 0.5:
        print("Diagnosis: 🔴 POSITIF DIABETES")
    else:
        print("Diagnosis: 🟢 NEGATIF DIABETES")

if __name__ == '__main__':
    # Matikan log GPU agar bersih
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    predict()   
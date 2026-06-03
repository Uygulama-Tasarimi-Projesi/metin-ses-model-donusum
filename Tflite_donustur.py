import tensorflow as tf
import numpy as np

# Tam yol — ses_saved_model nerede olursa olsun çalışır
SAVED_MODEL_DIR = r"C:\Users\asus\Desktop\3_sinif_bahar\Uygulama_tasarim\12.hafta\ses\ses_saved_model"
CIKTI_DOSYA   = r"C:\Users\asus\Desktop\3_sinif_bahar\Uygulama_tasarim\12.hafta\ses\en_iyi_ses_modeli.tflite"

print(">>> SavedModel yükleniyor...")
loaded = tf.saved_model.load(SAVED_MODEL_DIR)
infer  = loaded.signatures["serving_default"]

print("Input keys :", list(infer.structured_input_signature[1].keys()))
print("Output keys:", list(infer.structured_outputs.keys()))

print("\n>>> TFLite'a dönüştürülüyor...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([infer], loaded)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open(CIKTI_DOSYA, 'wb') as f:
    f.write(tflite_model)

print(f">>> Model boyutu: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Doğrulama
interpreter = tf.lite.Interpreter(model_path=CIKTI_DOSYA)
interpreter.allocate_tensors()
inp  = interpreter.get_input_details()
out  = interpreter.get_output_details()
print(f"Input shape : {inp[0]['shape']}")
print(f"Output shape: {out[0]['shape']}")
print("\n>>> TAMAMLANDI! Dosyayı android/src/main/assets/ klasörüne kopyala.")
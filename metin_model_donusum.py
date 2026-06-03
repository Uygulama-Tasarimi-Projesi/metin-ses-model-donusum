import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, LSTM, Dense
from tensorflow.keras.models import Model

# --- DOSYA YOLLARI ---
KERAS_MODEL = r"C://Users//asus//Desktop//3_sinif_bahar//Uygulama_tasarim//14.hafta//en_iyi_metin_modeli.keras"
CIKTI       = r"C://Users//asus//Desktop//3_sinif_bahar//Uygulama_tasarim//14.hafta//en_iyi_metin_modeli.tflite"

VOCAB_SIZE    = 10000
MAX_LENGTH    = 50
EMBEDDING_DIM = 128
NUM_CLASSES   = 7

print(">>> Orijinal mimari mobilde tahmin (inference) yapacak şekilde sadeleştiriliyor...")

# Giriş ve Gömme (Embedding)
input_layer = Input(shape=(MAX_LENGTH,), name="Metin_Giris_Katmani")
embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, name="Kelime_Gomme_Kati")(input_layer)

# CNN Bloğu (Aynı kalıyor)
conv_2 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', name="Conv1D_Bigram")(embedding_layer)
pool_2 = MaxPooling1D(pool_size=2, name="MaxPool_Bigram")(conv_2)

conv_3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="Conv1D_Trigram")(embedding_layer)
pool_3 = MaxPooling1D(pool_size=2, name="MaxPool_Trigram")(conv_3)

conv_4 = Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', name="Conv1D_4gram")(embedding_layer)
pool_4 = MaxPooling1D(pool_size=2, name="MaxPool_4gram")(conv_4)

cnn_features = Concatenate(axis=-1, name="N_Gram_Birlestirme")([pool_2, pool_3, pool_4])

# DİKKAT: Mobil tahmin için Dropout parametreleri tamamen kaldırıldı!
lstm_1 = LSTM(units=128, return_sequences=True, name="LSTM_Katmani_1")(cnn_features)
lstm_2 = LSTM(units=64, name="LSTM_Katmani_2")(lstm_1)

# Çıkış Katmanı (Dropout katmanı silindi)
output_layer = Dense(units=NUM_CLASSES, activation='softmax', name="Duygu_Cikisi_7_Sinif")(lstm_2)

fresh_model = Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Hibrit_Duygu_Modeli_Inference")

# Girdi tensörünü tetikle
_ = fresh_model(np.zeros((1, MAX_LENGTH), dtype=np.int32), training=False)

print("\n>>> Eğitilmiş .keras ağırlık dosyası yükleniyor...")
# Ağırlıkları yüklerken sildiğimiz dropout katmanlarından dolayı hata vermemesi için skip_mismatch kullanıyoruz
fresh_model.load_weights(KERAS_MODEL, skip_mismatch=True)
print(">>> Ağırlıklar başarıyla enjekte edildi.")

print("\n>>> Temizlenmiş model dışa aktarılıyor (Export)...")
SAVED_DIR = KERAS_MODEL.replace(".keras", "_saved_model")
fresh_model.export(SAVED_DIR)

print("\n>>> TensorFlow Lite (TFLite) Dönüşümü Başlatılıyor...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS 
]
converter._experimental_lower_tensor_list_ops = False

# HATA ÇÖZÜMÜ: TFLite'ın kaynak değişkenlerini (resource variables) okumasına izin veren kritik bayrak
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

with open(CIKTI, 'wb') as f:
    f.write(tflite_model)

print(f"\n>>> Model başarıyla TFLite formatına mühürlendi! Çıktı: {CIKTI}")
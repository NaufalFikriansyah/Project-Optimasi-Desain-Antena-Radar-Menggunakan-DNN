# deep_learning_antenna.py
# Skrip ini melatih model Deep Neural Network untuk memprediksi performa antena radar mikrostrip berbasis dataset hasil simulasi HFSS/CST

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === 1. Load dataset ===
data = pd.read_csv('dataset_antenna_radar10GHz_FR4_100.csv')

# === 2. Pisahkan fitur dan target ===
X = data[['Panjang_Patch_mm','Lebar_Patch_mm','Tebal_Substrat_mm','Epsilon_r']]
y = data[['Freq_Resonansi_GHz','Bandwidth_MHz','S11_dB','Gain_dBi','Efisiensi_persen']]

# === 3. Normalisasi fitur ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 5. Model DNN ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === 6. Training ===
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[es], verbose=1)

# === 7. Evaluasi ===
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"MAE rata-rata: {mae:.3f}")

# === 8. Simpan model ===
model.save('antenna_model.h5')
print("Model tersimpan sebagai 'antenna_model.h5'")

# === 9. Plot hasil training ===
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Grafik Konvergensi Model DNN')
plt.show()

# === 10. Prediksi contoh baru ===
contoh = np.array([[9.0, 12.0, 1.6, 2.2, 1, 0.3]])  # contoh parameter antena
contoh_scaled = scaler.transform(contoh)
prediksi = model.predict(contoh_scaled)
print("Prediksi performa antena (f_res, BW, S11, Gain, Efficiency):")
print(prediksi)

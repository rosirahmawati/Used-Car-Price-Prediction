import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle

print("1. Membaca dataset...")
df = pd.read_csv('train.csv')

print("2. Melakukan data pruning (Pembersihan Noise)...")

q_low = df['price'].quantile(0.01)
q_hi  = df['price'].quantile(0.99)
df_clean = df[(df['price'] < q_hi) & (df['price'] > q_low)].copy()

# Memisahkan Fitur (X) dan Target (y)
# Drop kolom id dan price (target)
X = df_clean.drop(columns=['id', 'price'])
y = df_clean['price']

# Mengidentifikasi kolom teks/kategorikal secara otomatis untuk CatBoost
cat_features_indices = np.where(X.dtypes == object)[0].tolist()

# Mengisi data kosong (missing values) pada kolom teks dengan string 'Unknown'
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(X[col].median())

print("3. Melatih model final dengan parameter terbaik...")

best_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features_indices,
    verbose=0 
)

# Proses Training
best_model.fit(X, y)
print("Model berhasil dilatih!")

print("4. Mengekspor model menjadi file biner...")
# Menyimpan model yang sudah pintar ke file pkl
with open('model_mobil.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Selesai! File 'model_mobil.pkl' telah siap di folder Anda.")
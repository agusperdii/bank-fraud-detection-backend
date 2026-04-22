# Fraud Detection API 🛡️

API ini digunakan untuk mendeteksi transaksi penipuan (fraud) menggunakan berbagai model Machine Learning seperti CatBoost, FT-Transformer, dan TabPFN.

## 📋 Prasyarat

- Python 3.9 atau lebih tinggi
- Pip (Python package manager)

## 🚀 Cara Instalasi

1.  **Clone atau unduh repository ini.**
2.  **Buat Virtual Environment (Opsional tapi disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    # atau
    venv\Scripts\activate     # Untuk Windows
    ```
3.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Cara Menjalankan

Jalankan server menggunakan Uvicorn:

```bash
python main.py
```
atau
```bash
uvicorn main:app --reload
```

Server akan berjalan di `http://localhost:8000`.

## 📖 Dokumentasi API

### 1. Health Check
Memeriksa status API dan model yang berhasil dimuat.

- **Endpoint:** `/health`
- **Metode:** `GET`

### 2. Prediksi Transaksi
Melakukan prediksi apakah sebuah transaksi merupakan fraud atau bukan.

- **Endpoint:** `/predict`
- **Metode:** `POST`
- **Format Body (JSON):**

```json
{
  "step": 1,
  "amount": 5000.0,
  "balanceDiffOrig": -5000.0,
  "balanceDiffDest": 5000.0,
  "destIsMerchant": 0,
  "senderTxnCount": 5,
  "receiverTxnCount": 2,
  "type_CASH_IN": 0,
  "type_CASH_OUT": 0,
  "type_DEBIT": 0,
  "type_PAYMENT": 0,
  "type_TRANSFER": 1
}
```

- **Respon:**
API akan mengembalikan list prediksi dari 3 model berbeda (CatBoost, FT-Transformer, dan TabPFN). Jika file model tidak ditemukan di server, API akan menggunakan logika *heuristic* sebagai cadangan (ditandai dengan `is_demo: true`).

## 📁 Struktur Model
Pastikan file berikut ada di direktori utama agar model dapat dimuat dengan benar:
- `catboost_optuna.pkl`
- `robust_scaler.pkl`
- `standard_scaler.pkl`
- `tabpfn_model.pkl`
- `ft_transformer_optuna/` (Folder berisi `model.ckpt` dll)

---
Dibuat dengan FastAPI.

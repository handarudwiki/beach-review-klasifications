# TourSense — Multi-Label Classifier Flask App

## Struktur Folder
```
flask_app/
├── app.py                     ← Flask backend
├── multilabel_nb_model.pkl    ← Model hasil training
├── label_binarizer.pkl        ← MultiLabelBinarizer
├── requirements.txt
├── templates/
│   └── index.html             ← UI utama
└── static/
    └── multilabel_nb_analysis.png
```

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Flask
```bash
python app.py
```

### 3. Buka browser
```
http://localhost:5000
```

## API Endpoints

### POST /predict
Prediksi satu atau beberapa review.

**Request:**
```json
{
  "reviews": ["Toilet sangat bersih dan pemandangan indah"]
}
```

**Response:**
```json
{
  "results": [
    {
      "review": "Toilet sangat bersih...",
      "cleaned": "toilet sih mandang indah",
      "labels": ["Fasilitas Kebersihan", "Keindahan"],
      "label_details": [
        {
          "name": "Fasilitas Kebersihan",
          "predicted": true,
          "confidence": 0.89,
          "emoji": "🧹",
          "color": "#2ECC71",
          "desc": "Kebersihan & fasilitas umum"
        }
        ...
      ]
    }
  ]
}
```

## Labels
| Label | Emoji | Deskripsi |
|---|---|---|
| Fasilitas Kebersihan | 🧹 | Kebersihan & fasilitas umum |
| Keindahan | 🌅 | Keindahan visual & estetika |
| Alam | 🌿 | Flora, fauna & lingkungan alam |
| Aksesibilitas | 🛣️ | Kemudahan akses & transportasi |
| Others | 💬 | Harga, pelayanan & lainnya |

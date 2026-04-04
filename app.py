"""
Flask API + UI for Multi-Label Tourism Review Classifier
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

# ── Load model & binarizer ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "multilabel_nb_model.pkl"))
mlb   = joblib.load(os.path.join(BASE_DIR, "label_binarizer.pkl"))

LABEL_NAMES = ["Fasilitas Kebersihan", "Keindahan", "Alam", "Aksesibilitas", "Others"]

LABEL_META = {
    "Fasilitas Kebersihan": {"emoji": "🧹", "color": "#2ECC71", "desc": "Kebersihan & fasilitas umum"},
    "Keindahan":            {"emoji": "🌅", "color": "#E74C3C", "desc": "Keindahan visual & estetika"},
    "Alam":                 {"emoji": "🌿", "color": "#27AE60", "desc": "Flora, fauna & lingkungan alam"},
    "Aksesibilitas":        {"emoji": "🛣️",  "color": "#3498DB", "desc": "Kemudahan akses & transportasi"},
    "Others":               {"emoji": "💬", "color": "#9B59B6", "desc": "Harga, pelayanan & lainnya"},
}

# ── Indonesian stopwords ───────────────────────────────────────────────
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "adalah", "ada", "pada", "juga", "saya", "kami", "kita", "mereka",
    "sangat", "sudah", "bisa", "tidak", "ya", "atau", "akan", "lebih",
    "satu", "dua", "tiga", "ber", "ter", "me", "per", "an", "kan",
    "lagi", "jadi", "tapi", "namun", "karena", "kalau", "jika", "maka",
    "agar", "supaya", "hingga", "serta", "maupun", "bahwa", "sehingga",
    "dalam", "oleh", "atas", "bawah", "antara", "tanpa", "setelah",
    "sebelum", "selama", "seperti", "sama", "semua", "setiap", "masih",
    "belum", "pun", "nih", "sih", "deh", "dong", "lho",
    "nya", "mu", "ku", "punya", "milik", "bagi", "para",
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 2]
    cleaned = []
    for token in tokens:
        for prefix in ["me", "ber", "ter", "pe", "ke", "se"]:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                token = token[len(prefix):]
                break
        for suffix in ["kan", "an", "nya", "i"]:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break
        if len(token) > 2:
            cleaned.append(token)
    return " ".join(cleaned)


# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", label_meta=LABEL_META)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    reviews_raw = data.get("reviews", [])

    if not reviews_raw:
        return jsonify({"error": "No reviews provided"}), 400

    results = []
    for review in reviews_raw:
        if not review.strip():
            continue
        cleaned   = preprocess_text(review)
        pred_bin  = model.predict([cleaned])
        pred_prob = None

        # Get probability scores per label (OneVsRest exposes estimators)
        try:
            probs = []
            for est in model.named_steps["clf"].estimators_:
                tfidf_feat = model.named_steps["tfidf"].transform([cleaned])
                p = est.predict_proba(tfidf_feat)[0]
                probs.append(round(float(max(p)), 3))
            pred_prob = probs
        except Exception:
            pred_prob = [None] * len(LABEL_NAMES)

        pred_labels = mlb.inverse_transform(pred_bin)[0]
        pred_labels = list(pred_labels) if pred_labels else ["Others"]

        label_details = []
        for i, label in enumerate(LABEL_NAMES):
            label_details.append({
                "name":       label,
                "predicted":  label in pred_labels,
                "confidence": pred_prob[i] if pred_prob else None,
                "emoji":      LABEL_META[label]["emoji"],
                "color":      LABEL_META[label]["color"],
                "desc":       LABEL_META[label]["desc"],
            })

        results.append({
            "review":       review,
            "cleaned":      cleaned,
            "labels":       pred_labels,
            "label_details": label_details,
        })

    return jsonify({"results": results})


@app.route("/batch", methods=["POST"])
def batch():
    """Batch prediction from textarea (newline-separated)"""
    data    = request.get_json()
    raw_text = data.get("text", "")
    reviews  = [r.strip() for r in raw_text.split("\n") if r.strip()]
    if not reviews:
        return jsonify({"error": "No reviews found"}), 400

    return predict_list(reviews)

def predict_list(reviews_raw):
    results = []
    for review in reviews_raw:
        cleaned  = preprocess_text(review)
        pred_bin = model.predict([cleaned])
        pred_labels = mlb.inverse_transform(pred_bin)[0]
        pred_labels = list(pred_labels) if pred_labels else ["Others"]

        probs = []
        try:
            for est in model.named_steps["clf"].estimators_:
                tfidf_feat = model.named_steps["tfidf"].transform([cleaned])
                p = est.predict_proba(tfidf_feat)[0]
                probs.append(round(float(max(p)), 3))
        except Exception:
            probs = [None] * len(LABEL_NAMES)

        label_details = []
        for i, label in enumerate(LABEL_NAMES):
            label_details.append({
                "name":       label,
                "predicted":  label in pred_labels,
                "confidence": probs[i],
                "emoji":      LABEL_META[label]["emoji"],
                "color":      LABEL_META[label]["color"],
                "desc":       LABEL_META[label]["desc"],
            })

        results.append({
            "review":        review,
            "cleaned":       cleaned,
            "labels":        pred_labels,
            "label_details": label_details,
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

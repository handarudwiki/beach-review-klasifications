"""
Flask API + UI for Multi-Label Tourism Review Classifier
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)

# ── Load model & binarizer ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "multilabel_nb_model.pkl"))

LABEL_NAMES = ["Fasilitas", "Kebersihan", "Keindahan Alam", "Aksesibilitas", "Others"]

LABEL_META = {
    "Fasilitas":      {"emoji": "🏢", "color": "#2ECC71", "desc": "Fasilitas umum & amenitas"},
    "Kebersihan":     {"emoji": "🧹", "color": "#E74C3C", "desc": "Kebersihan & sanitasi"},
    "Keindahan Alam": {"emoji": "🌿", "color": "#27AE60", "desc": "Keindahan alam & pemandangan"},
    "Aksesibilitas":  {"emoji": "🛣️",  "color": "#3498DB", "desc": "Kemudahan akses & transportasi"},
    "Others":         {"emoji": "💬", "color": "#9B59B6", "desc": "Harga, pelayanan & lainnya"},
}

# ── Indonesian stopwords ───────────────────────────────────────────────

EXTRA_STOPWORDS = {
    "nih", "sih", "deh", "dong", "lho", "tuh", "yuk", "wah",
    "banget", "bgt", "udah", "udh", "gak", "nggak", "ga",
    "nya", "mu", "ku", "tp", "yg", "dgn", "utk", "jg", "krn",
    "klo", "kl", "emg", "emang", "kayak", "kaya", "aja", "doang",
}

_stemmer_factory = StemmerFactory()
_stemmer         = _stemmer_factory.create_stemmer()

_sw_factory      = StopWordRemoverFactory()
_sw_remover      = _sw_factory.create_stop_word_remover()

def translate_to_indonesian(text):
    """Translate English to Indonesian. Skip if already Indonesian or other language."""
    try:
        lang = detect(text)
        if lang != "en":
            print("Lang : ", lang)
            print(f"Skipping translation for {lang}: {text}")
            return text
        translated = GoogleTranslator(source="en", target="id").translate(text)
        print("translated : ", translated)
        return translated if translated else text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# ── Atomic preprocessing step functions ──────────────────────────────
def step_translate(text):
    return translate_to_indonesian(text)

def step_lowercase(text):
    return text.lower()

def step_remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)

def step_remove_html(text):
    return re.sub(r"<[^>]+>", "", text)

def step_remove_special_chars(text):
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def step_remove_stopwords(text):
    return _sw_remover.remove(text)

def step_stem(text):
    return _stemmer.stem(text)

def step_filter_tokens(text):
    tokens = [t for t in text.split() if len(t) > 2 and t not in EXTRA_STOPWORDS]
    return " ".join(tokens)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = step_translate(text)
    text = step_lowercase(text)
    text = step_remove_urls(text)
    text = step_remove_html(text)
    text = step_remove_special_chars(text)
    text = step_remove_stopwords(text)
    text = step_stem(text)
    text = step_filter_tokens(text)

    return text


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

        pred_labels = [LABEL_NAMES[j] for j, v in enumerate(pred_bin[0]) if v == 1]
        pred_labels = pred_labels if pred_labels else ["Others"]

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
        pred_labels = [LABEL_NAMES[j] for j, v in enumerate(pred_bin[0]) if v == 1]
        pred_labels = pred_labels if pred_labels else ["Others"]

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


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

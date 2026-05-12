import os
import whisper
import joblib
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ------------------- CONFIG -------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- APP INIT -------------------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------- LOAD MODELS -------------------
print("🔄 Loading ML models...")

whisper_model = whisper.load_model("base")
text_model = joblib.load("model/wartime_distress_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

print("✅ Models loaded successfully")

# ------------------- KEYWORD SYSTEM -------------------

# CRITICAL keywords — always DISTRESS regardless of context
CRITICAL_KEYWORDS = [
    # Weapons / Ordnance
    r"\bbombs?\b", r"\bshelling\b", r"\bartillery\b", r"\bmissile[s]?\b",
    r"\brocket[s]?\b", r"\bmortar[s]?\b", r"\bgrenade[s]?\b", r"\bexplosive[s]?\b",
    r"\blandmine[s]?\b", r"\bied\b", r"\bcluster.?bomb\b", r"\bsuicide.?bomb\b",
    r"\bsniper[s]?\b", r"\bgunfire\b", r"\bfiring\b", r"\bshrapnel\b",
    r"\bbarrage\b", r"\bbullet[s]?\b", r"\bammunition\b", r"\bdetonation\b",
    r"\bphosphorus\b", r"\bincendiary\b", r"\bdirty.?bomb\b", r"\bnuclear.?strike\b",
    r"\bnuclear.?detonation\b", r"\ballistic.?missile\b", r"\btactical.?nuclear\b",

    # Threat / Attack
    r"\battack(ing|ed)?\b", r"\bambush\b", r"\bassault\b",
    r"\benemy.?(troops|forces|fire|ground)\b", r"\bhostile[s]?\b",
    r"\binvasion\b", r"\bparatroopers?\b", r"\bsniping\b",

    # Emergency / Distress
    r"\bsos\b", r"\bmayday\b", r"\bhelp.?me\b", r"\bsave.?us\b",
    r"\bwe.?are.?trapped\b", r"\btrapped.?under\b", r"\bstuck.?in\b",
    r"\bwe.?need.?help\b", r"\bsend.?help\b", r"\bsend.?ambulance\b",
    r"\burgent(ly)?\b", r"\bemergency\b", r"\bevacuat(e|ion)\b",
    r"\bextract(ion)?\b", r"\bcasualt(y|ies)\b", r"\bwounded\b",
    r"\binjur(ed|ies)\b", r"\bblood(ing)?\b", r"\bmedic\b",
    r"\bcritical.?condition\b", r"\bsevere.?wound\b",

    # Radiation / CBRN
    r"\bradiation\b", r"\bnuclear\b", r"\bfallout\b", r"\bhazmat\b",
    r"\bchemical.?attack\b", r"\bbiological.?weapon\b", r"\bgas.?attack\b",

    # Hindi equivalents
    r"\bमदद\b", r"\bआपातकाल\b", r"\bहमला\b", r"\bबम\b", r"\bगोलीबारी\b",

    # Russian equivalents
    r"\bпомогите\b", r"\bпомощь\b", r"\bбомба\b", r"\bатак\b",

    # Japanese equivalents
    r"\b助けて\b", r"\b爆弾\b", r"\b攻撃\b",
]

# SAFE context overrides — if these appear, neutralize certain terms
SAFE_OVERRIDES = [
    r"\bdefused?\b", r"\bsafely.?removed\b", r"\bdisposed?\b",
    r"\bordnance.?disposal\b", r"\bbomb.?squad\b", r"\bcleared.?the.?sector\b",
    r"\bcease.?fire\b", r"\bceasefire\b", r"\breactor.?offline\b",
    r"\bon.?maintenance\b", r"\broutine.?patrol\b", r"\bsmoke.?rounds\b",
    r"\boutine\b",
]

# Terms that ALONE (without distress context) should NOT trigger critical
AMBIGUOUS_TERMS = {
    r"\bnuclear\b", r"\bartillery\b", r"\bbomb\b", r"\bmines?\b"
}

def check_keywords(text):
    """
    Returns (is_critical, matched_keyword)
    Uses three-tier logic:
    1. MARS codename — always CRITICAL, no exceptions
    2. Check for safe-override context
    3. Check critical keywords
    4. Ambiguous terms get context check
    """
    text_lower = text.lower()

    # 🚨 PROJECT MARS — highest priority override, always CRITICAL
    if "mars" in text_lower:
        return True, "MARS"

    has_safe_context = any(re.search(p, text_lower) for p in SAFE_OVERRIDES)

    # Check critical keywords
    for pattern in CRITICAL_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched = pattern

            # If it's an ambiguous term, only override if there's NO safe context
            is_ambiguous = any(re.search(p, text_lower, re.IGNORECASE) for p in AMBIGUOUS_TERMS)
            if is_ambiguous and has_safe_context:
                continue  # Safe context neutralizes the ambiguous term

            # For unambiguous critical keywords, always distress
            return True, matched

    return False, None


def classify_text(text):
    """
    Three-stage classification:
    1. Keyword override (highest priority)
    2. ML model vote
    3. Confidence-adjusted final decision
    """
    if not text or not text.strip():
        return "NORMAL", 0.5

    # Stage 1: Keyword check
    is_critical, matched = check_keywords(text)

    # Stage 2: ML model
    vec = vectorizer.transform([text])
    pred = text_model.predict(vec)[0]
    prob_array = text_model.predict_proba(vec)[0]
    ml_distress_prob = float(prob_array[1]) if len(prob_array) > 1 else 0.0

    # Stage 3: Decision fusion
    if is_critical:
        # MARS codename — always returns CRITICAL status
        if matched == "MARS":
            return "CRITICAL", 1.0
        # Other keywords — DISTRESS with high confidence
        final_confidence = max(0.90, ml_distress_prob)
        return "DISTRESS", round(final_confidence, 2)

    if pred == 1 and ml_distress_prob >= 0.60:
        return "DISTRESS", round(ml_distress_prob, 2)

    if pred == 0 and ml_distress_prob <= 0.40:
        return "NORMAL", round(1 - ml_distress_prob, 2)

    # Borderline: trust keyword absence, return NORMAL with low confidence
    return "NORMAL", round(1 - ml_distress_prob, 2)


# ------------------- UTILS -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    prediction, confidence = classify_text(text)
    return jsonify({"status": prediction, "confidence": confidence})


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported audio format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        transcription_result = whisper_model.transcribe(filepath)
        text = transcription_result["text"].strip()
        prediction, confidence = classify_text(text)
        return jsonify({
            "transcription": text,
            "prediction": prediction,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
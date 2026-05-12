"""
train_model.py — Upgraded Wartime Distress Signal Classifier
Uses TF-IDF with n-grams + Logistic Regression with class balancing.
Run this once to regenerate model/vectorizer.pkl files.
"""

import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
import numpy as np

os.makedirs("model", exist_ok=True)

# ============================================================
# DATASET — Carefully curated wartime distress examples
# ============================================================
# Key principle: dangerous WORDS in SAFE CONTEXT = label 0
#                dangerous WORDS in DISTRESS CONTEXT = label 1
# ============================================================

data = [
    # ---- DISTRESS SIGNALS (label = 1) ----
    # Direct calls for help
    ("help we are trapped", 1),
    ("help me please we are under attack", 1),
    ("send help immediately civilians trapped", 1),
    ("mayday mayday requesting urgent assistance", 1),
    ("sos we need immediate rescue", 1),
    ("we are surrounded please send help now", 1),
    ("urgent help needed people are dying here", 1),

    # Medical emergencies
    ("need medical assistance urgently", 1),
    ("send ambulance immediately multiple casualties", 1),
    ("medic needed for a severe chest wound", 1),
    ("shrapnel injuries to multiple children near the school", 1),
    ("emergency medical evacuation needed for two civilians", 1),
    ("wounded soldiers require immediate extraction", 1),
    ("critical injuries from the blast send a medic now", 1),
    ("civilian bleeding heavily from gunshot wound", 1),
    ("man down multiple injuries need medevac now", 1),
    ("three people badly injured need ambulance at checkpoint", 1),
    ("soldier has lost consciousness from blast injuries", 1),
    ("severe burn wounds from incendiary attack need treatment", 1),

    # Active combat / incoming fire
    ("enemy firing near village", 1),
    ("sniper fire pinning us down on the main road", 1),
    ("heavy artillery shelling started near our sector", 1),
    ("incoming artillery barrage seek cover now", 1),
    ("enemy ground troops entering the western outskirts", 1),
    ("missile siren sounding please seek immediate shelter", 1),
    ("tank fire has destroyed the local water tower", 1),
    ("rocket attack on civilian housing block in progress", 1),
    ("mortar rounds landing close to refugee camp", 1),
    ("gunfire erupting in the northern district right now", 1),
    ("heavy shelling of the town center is ongoing", 1),
    ("drones circling and launching strikes on vehicles", 1),
    ("we are under sustained machine gun fire", 1),
    ("enemy ambush on the convoy at grid 447", 1),
    ("hostile forces breaching the perimeter now", 1),
    ("paratroopers landing in the fields behind us", 1),
    ("enemy tanks advancing rapidly on the highway", 1),
    ("jets conducting airstrikes over the eastern sector", 1),
    ("gunship helicopter firing on civilians in the square", 1),

    # Explosives and ordnance threats
    ("unexploded ordnance found in the residential garden", 1),
    ("landmines spotted on the dirt path near the farm", 1),
    ("improvised explosive device detonated on the convoy", 1),
    ("suicide bomb attack at the southern checkpoint", 1),
    ("ballistic missile launch detected heading our direction", 1),
    ("suspected pipe bomb found near the water station", 1),
    ("cluster bomb detonating across the field near us", 1),
    ("car bomb exploded near the market this morning", 1),
    ("grenade thrown into the shelter we need help now", 1),
    ("building collapsed from bomb strike we are buried", 1),
    ("trapped under a collapsed building after the missile strike", 1),
    ("phosphorus bomb burns reported at the clinic", 1),
    ("dirty bomb threat detected in the subway tunnels", 1),
    ("unexploded cluster bomb in the middle of the road", 1),

    # CBRN threats
    ("detected high radiation levels near the blast site", 1),
    ("nuclear strike warning issued for the city evacuate now", 1),
    ("white phosphorus detected in the northern district", 1),
    ("chemical gas attack on the village evacuation needed", 1),
    ("biological agent released in the market seek shelter", 1),
    ("nuclear detonation confirmed casualties in the thousands", 1),

    # Structural / environment collapse
    ("building is on fire people are still inside", 1),
    ("bridge destroyed by airstrike we have no escape", 1),
    ("flooding from dam breach civilians in danger now", 1),
    ("road is blocked by rubble we cannot get through", 1),
    ("power plant on fire releasing toxic fumes evacuate", 1),

    # Multilingual distress
    ("मदद करें हम फंसे हुए हैं", 1),
    ("तुरंत चिकित्सा सहायता चाहिए", 1),
    ("गांव के पास दुश्मन की फायरिंग हो रही है", 1),
    ("तुरंत एंबुलेंस भेजें", 1),
    ("हमारे क्षेत्र के पास भारी तोपखाना गोलाबारी शुरू हो गई है", 1),
    ("मुख्य सड़क पर स्नाइपर फायर हमें रोक रहा है", 1),
    ("помогите мы в ловушке", 1),
    ("срочно нужна медицинская помощь", 1),
    ("вражеский обстрел возле деревни", 1),
    ("немедленно отправьте скорую помощь", 1),
    ("начался тяжелый артиллерийский обстрел рядом с нашим сектором", 1),
    ("助けて 私たちは閉じ込められています", 1),
    ("緊急の医療支援が必要です", 1),
    ("村の近くで敵の発砲があります", 1),
    ("すぐに救急車を送ってください", 1),

    # ---- NORMAL / SAFE SIGNALS (label = 0) ----
    # All clear / safe
    ("we are safe and fine", 0),
    ("having dinner at home", 0),
    ("normal patrol completed no incidents", 0),
    ("weather is calm and clear today", 0),
    ("the humanitarian corridor is currently open", 0),
    ("we have enough food and water for three days", 0),
    ("staying in the basement until the morning as precaution", 0),
    ("the local market is selling bread today business as usual", 0),
    ("we reached the designated safe zone successfully", 0),
    ("patrol units report no enemy activity in sector 4", 0),
    ("all family members are accounted for in the shelter", 0),
    ("the ceasefire is holding and the area is quiet", 0),
    ("observing ceasefire from our position no movement", 0),
    ("evacuation bus has arrived at the checkpoint safely", 0),
    ("distributing blankets to the families in the bunker", 0),
    ("the radio signal is clear and communication is stable", 0),
    ("everything is fine here no threats detected", 0),
    ("I am safe please do not worry about me", 0),
    ("all units report green status operations normal", 0),
    ("the generator is working and we have stable power", 0),

    # Safe uses of dangerous-sounding words (KEY for training)
    ("the bomb squad has successfully defused the device", 0),
    ("ordnance disposal team is clearing the sector safely", 0),
    ("the nuclear reactor is offline for routine maintenance", 0),
    ("strategic nuclear bombers are on scheduled patrol", 0),
    ("the artillery battery has ceased firing for now", 0),
    ("military transport carrying medical supplies arrived", 0),
    ("checking the perimeter for anti-tank mines as precaution", 0),
    ("the bomb shelter air filters are working fine", 0),
    ("loading the artillery pieces with smoke rounds for drill", 0),
    ("nuclear power plant cooling systems are fully stable", 0),
    ("the old landmine was safely removed by the EOD team", 0),
    ("missile test was conducted successfully over the ocean", 0),
    ("artillery unit completed training exercise without incident", 0),
    ("the sniper range was cleared after the training session", 0),
    ("the drone returned safely after the surveillance mission", 0),
    ("rocket test launch was a success at the facility", 0),
    ("the grenade range is closed after the training exercise", 0),
    ("the old bunker has been repurposed as a storage room", 0),
    ("the deactivated bomb is now in a museum display", 0),
    ("explosive residue detected but the area was contained", 0),

    # Everyday life
    ("i am eating lunch peacefully", 0),
    ("i am sleeping in my room", 0),
    ("i am happy today", 0),
    ("i am sad today", 0),
    ("i am angry about the traffic", 0),
    ("i am tired after a long day of work", 0),
    ("i am bored waiting for the bus", 0),
    ("i went shopping at the supermarket", 0),
    ("i called my family today and they are well", 0),
    ("my phone battery is dead charging now", 0),

    # Multilingual safe
    ("हम सुरक्षित और ठीक हैं", 0),
    ("घर पर खाना खा रहे हैं", 0),
    ("सामान्य गश्त पूरी हुई", 0),
    ("आज मौसम शांत है", 0),
    ("मैं शांतिपूर्वक खाना खा रहा हूँ", 0),
    ("मैं आज खुश हूँ", 0),
    ("мы в безопасности и всё хорошо", 0),
    ("ужинаем дома всё спокойно", 0),
    ("обычный патруль завершен без происшествий", 0),
    ("я спокойно обедаю дома", 0),
    ("私たちは安全で問題ありません", 0),
    ("家で夕食を食べています", 0),
    ("通常のパトロールが完了しました", 0),
    ("今日は天気が穏やかです", 0),
    ("今日は幸せです", 0),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

print(f"📊 Dataset: {len(texts)} samples | {sum(labels)} distress | {len(labels)-sum(labels)} normal")

# ============================================================
# VECTORIZER — Upgraded with bigrams and char n-grams
# ============================================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),       # unigrams, bigrams, trigrams
    max_features=8000,
    sublinear_tf=True,        # log-scale TF to reduce impact of frequent terms
    min_df=1,
    analyzer="word",
    strip_accents="unicode",
)

X = vectorizer.fit_transform(texts)
y = np.array(labels)

# ============================================================
# MODEL — Logistic Regression with class balancing
# ============================================================
model = LogisticRegression(
    C=1.5,
    max_iter=1000,
    class_weight="balanced",  # handles class imbalance
    solver="lbfgs",
)
model.fit(X, y)

# ============================================================
# EVALUATION
# ============================================================
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"\n✅ Cross-val F1 scores: {cv_scores.round(3)}")
print(f"   Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_eval = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", solver="lbfgs")
model_eval.fit(X_train, y_train)
y_pred = model_eval.predict(X_test)
print(f"\n📋 Classification Report (hold-out 20%):\n")
print(classification_report(y_test, y_pred, target_names=["NORMAL", "DISTRESS"]))

# Save final model trained on full data
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(model, "model/wartime_distress_model.pkl")
print("\n✅ Model & vectorizer saved successfully to model/")
print("   Run app.py to start the server.")
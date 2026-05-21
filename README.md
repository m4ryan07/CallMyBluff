````md
🃏 CallMyBluff — AI Text Detector

Detects AI-generated text by analyzing how it's written, not what it's about.  
Pure stylometric feature engineering — no bag-of-words shortcuts.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)

---

📖 What is CallMyBluff?

Most AI detectors count words.  
CallMyBluff reads writing style.

Just like a poker player's tell reveals their hand, every writer has unconscious stylistic patterns — sentence rhythm, vocabulary habits, punctuation instincts — that are nearly impossible to fake.

CallMyBluff measures these signals to determine whether a human or an AI is holding the pen.

---

🧠 Stylometric Features

• Burstiness  
  Human writing mixes short punchy sentences with long complex ones.  
  AI writing tends to maintain uniform sentence lengths.

• Contraction Rate  
  Humans naturally write things like:
  don't, I've, they're

  AI often defaults to:
  do not, I have, they are

• Vocabulary Diversity (TTR)  
  Humans repeat words organically.  
  AI often produces artificially high vocabulary variation.

• Rhetorical Questions  
  Humans frequently ask things like:
  "Have you ever thought about...?"

  AI rarely does.

• Comma Rate  
  Humans use commas naturally for clauses.  
  AI often overuses commas in list-like structures.

• Punctuation Density  
  Human writing contains emotional punctuation:
  dashes, ellipses, pauses

  AI writing is usually cleaner and more sterile.

---

🏗️ Project Structure

```text
CallMyBluff/
│
├── app.py
├── ingest.py
├── generate_ai_samples.py
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── static/
│   ├── style.css
│   └── app.js
│
├── templates/
│   └── index.html
│
├── models/
│
└── data/
    ├── processed/
    └── plots/
````

app.py
Flask backend and API routes

ingest.py
Dataset loading and train/test splitting

generate_ai_samples.py
Groq API script for AI-generated essay creation

features.py
Stylometric feature extraction logic

train.py
Machine learning training pipeline

evaluate.py
ROC curve, confusion matrix, evaluation metrics

style.css
Dark poker-inspired frontend styling

app.js
AJAX requests, animations, dynamic feature bars

index.html
Frontend template rendered with Flask

---

⚙️ Local Setup

1️⃣ Clone the Repository

```bash
git clone https://github.com/m4ryan07/CallMyBluff.git
cd CallMyBluff
```

2️⃣ Create Virtual Environment

macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Download NLTK Data

```bash
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
"
```

5️⃣ Run the Application

```bash
python3 app.py
```

Open in browser:

```text
http://localhost:5050
```

Note:
The trained model is already included in the models/ directory.
Retraining is optional unless you want to use your own dataset.

---

🔬 ML Pipeline

```text
Raw Text
   ↓
Stylometric Feature Extraction
   ↓
MaxAbsScaler Normalization
   ↓
Logistic Regression Classifier
   ↓
Verdict + Confidence Score
```

---

📈 Models Benchmarked

Logistic Regression
• F1 Weighted: 1.000
• ROC-AUC: 1.000

Complement Naive Bayes
• F1 Weighted: 0.950
• ROC-AUC: 0.997

Random Forest
• F1 Weighted: 1.000
• ROC-AUC: 1.000

Logistic Regression was selected because its coefficients are interpretable.
The model clearly shows which stylometric features influence Human vs AI classification.

---

📊 Evaluation

Accuracy: 100%
ROC-AUC: 1.000
False Positive Rate: 0.00%
False Negative Rate: 0.00%

Training Dataset:
• 150 human essays from the DAIGT dataset
• 150 Groq LLaMA 3 generated essays on matching prompts

---

📸 Screenshots

uploaded with the files
---


🛠️ Tech Stack

ML & Feature Engineering
Scikit-Learn, NLTK, NumPy, Pandas

Backend
Flask

Frontend
HTML, CSS, JavaScript

Dataset
DAIGT + Groq LLaMA 3

Serialization
Joblib

---

👨‍💻 About

Built as a portfolio project demonstrating:

• Dataset construction
• Stylometric feature engineering
• Machine learning training
• Model evaluation
• Flask backend integration
• Frontend development
• End-to-end deployment workflow

---

"The best bluff is the one you don't know you're making."

---

📦 Push to GitHub

```bash
git add README.md
git commit -m "docs: add README"
git push origin main
```

```
```


BY M ARYAN

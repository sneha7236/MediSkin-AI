# MediSkin AI

**AI-powered skin condition identification and healthcare recommendation platform** — a final-year major project.

> ⚠️ **Medical disclaimer.** This result is for preliminary informational purposes only and is not a medical diagnosis. Please consult a qualified healthcare professional for proper evaluation and treatment.

---

## 1. Project overview

MediSkin AI lets a signed-in user upload a photo of a skin area. The image is validated and preprocessed, a trained deep-learning model returns a probability for each supported skin-condition class, and the app presents the top suggestion with its confidence, an informational triage indicator, educational information, suggested next steps, a hospital finder, a downloadable PDF report and a stored history.

It is an **educational screening aid, not a diagnostic tool.** Confidence is the model's own probability, never medical certainty.

> **Important — no trained model ships with this repository.** There is no dataset or model in the repo, and no accuracy figure is claimed anywhere. The full training pipeline is included; until you train (or install) a model the app starts normally, explains that the model is missing, and disables analysis. Model analytics show real numbers only after a training run writes `models/model_metrics.json`.

## 2. Features

| Area | What is implemented |
|---|---|
| Accounts | Registration (email format, password strength, duplicate-email checks), login, logout, profile, name update, password change, session timeout, login throttling |
| Analysis | JPG/JPEG/PNG upload, validation (type, real image content, size, emptiness, corruption, minimum size), preview, EXIF-aware preprocessing, prediction, loading states |
| Results | Predicted condition, confidence band (High ≥ 0.75, Moderate 0.50–0.74, Low < 0.50), triage indicator, top-3 probabilities, condition education, care and "when to seek advice" guidance |
| Explainability | Grad-CAM (original / heatmap / overlay) labelled "not a clinical interpretation"; hidden automatically if the model has no suitable convolutional layer |
| History | Per-user prediction history, view stored result, download/create PDF, delete with confirmation |
| Reports | PDF reports (ReportLab) tracked in the database in `reports/generated/` with unique filenames and path-traversal protection |
| Hospitals | Provider interface (`HospitalProvider`) reading real entries from `data/metadata/hospitals.json`; distance only when coordinates exist; map-search link; **no hospital data is invented** |
| Feedback | Star rating, comment, helpful / not helpful, optionally tied to an analysis |
| Analytics | Accuracy, macro precision/recall/F1, per-class table, confusion matrix, training curves — all read from files written by training |
| Security | bcrypt password hashes, input validation, upload validation, safe file paths, upload size limit, secrets only via `.env`, no secrets or passwords logged |
| UI | Custom CSS healthcare theme, sidebar navigation, cards, badges, empty/loading/error/success states, responsive layout, visible disclaimer on every page |

## 3. Technology stack

Python · Streamlit (+ custom CSS) · TensorFlow/Keras (MobileNetV2 transfer learning or a custom CNN, Batch Normalization, Adam, augmentation) · SQLite via SQLAlchemy 2 · Pillow, NumPy · pandas, scikit-learn, Matplotlib · ReportLab · bcrypt · python-dotenv · pytest (+ optional Playwright for the browser test).

OpenCV is not used: Pillow covers every image operation this project needs.

## 4. Architecture

```
Browser ── Streamlit pages (UI only) ──┬─ modules/      authentication, history, disease_info,
                                       │               hospital_recommendation, report_generator,
   app.py: config, session guard,      │               feedback, analytics   (business logic)
   navigation                          ├─ prediction/   image_processor → predictor → confidence,
                                       │               explainability (Grad-CAM)
                                       ├─ database/     SQLAlchemy engine, models, CRUD
                                       └─ utils/        validators, security, logger, helpers, ui

Training (offline CLI):  data/raw → training.preprocess → training.train → models/
                         (clean, lesion-aware split)   (augment, CNN/BN/Adam, evaluate, save)
```

Rules followed: pages never touch the database or the model directly; prediction code never imports Streamlit; configuration lives in `config/` and `.env`; the model is loaded once per process (`st.cache_resource`, reloaded automatically if the model file changes).

## 5. Folder structure

```
MediSkin-AI/
├── app.py                      # entry point: page config, startup, session guard, navigation
├── requirements.txt
├── README.md
├── .env.example                # copy to .env
├── .gitignore
├── .streamlit/config.toml      # theme + upload ceiling
├── config/                     # config.py (env-driven), constants.py (thresholds, text)
├── data/
│   ├── raw/                    # your dataset: raw/<class_slug>/*.jpg
│   ├── processed/              # generated train/val/test PNGs
│   └── metadata/               # dataset_manifest.csv, hospitals.json
├── models/                     # skin_model.keras, class_names.json, model_metrics.json (generated)
├── training/                   # organize_ham10000, preprocess, augmentation, train, evaluate
├── prediction/                 # image_processor, predictor, confidence, explainability
├── database/                   # database.py, models.py, crud.py
├── modules/                    # authentication, disease_info, hospital_recommendation,
│                               # report_generator, feedback, history, analytics
├── pages/                      # auth, home, analyze, results, history, reports, hospitals,
│                               # analytics, feedback, profile, about
├── utils/                      # validators, security, helpers, logger, ui
├── assets/                     # styles/style.css, images/ (logo)
├── reports/generated/          # generated PDFs
├── logs/                       # rotating application log
└── tests/                      # unit tests + tests/e2e/run_e2e.py (browser test)
```

The SQLite database `medi_skin.db` is created in the project root on first start.

## 6. Installation

Requires Python 3.10+ (developed and tested with Python 3.13 and TensorFlow 2.21).

```bash
git clone <your-repository-url> MediSkin-AI
cd MediSkin-AI
```

## 7. Virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

## 8. Install requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

## 9. Dataset setup

No dataset is bundled. The default class set matches the seven **HAM10000** diagnostic categories (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`), which have built-in education content. Any other folder-per-class dataset also works; unknown classes get a safe generic information card.

1. Download HAM10000 yourself (Harvard Dataverse / Kaggle / ISIC archive) and unzip it so the image folders and `HAM10000_metadata.csv` are under one folder. Respect the dataset licence (CC BY-NC 4.0 — non-commercial).
2. Arrange it into `data/raw/<class>/`:

```bash
python -m training.organize_ham10000 --source /path/to/HAM10000
```

   Files are named `<lesion_id>__<image_id>.jpg`. HAM10000 contains several photos of the same lesion; the split in the next step keeps every photo of a lesion in the same partition so test metrics are not inflated by near-duplicates.

   *Using another dataset?* Place images in `data/raw/<class_name>/*.jpg|jpeg|png` yourself (minimum 6 valid images per class, at least 2 classes).

3. Clean, split (70/15/15, stratified) and resize:

```bash
python -m training.preprocess
```

   Unreadable, tiny and byte-identical duplicate images are removed. Output goes to `data/processed/` with a manifest in `data/metadata/`.

## 10. Model training

```bash
# Recommended: MobileNetV2 transfer learning (downloads ImageNet weights once; needs internet)
python -m training.train --arch mobilenetv2 --epochs 15 --fine-tune-epochs 10

# No internet / quick experiment: small custom CNN with Batch Normalization
python -m training.train --arch cnn --epochs 30
```

Training uses augmentation (training split only), class-weighted loss for imbalanced classes, Adam, early stopping and learning-rate reduction, then evaluates on the held-out test split. It writes:

- `models/skin_model.keras`
- `models/class_names.json`
- `models/model_metrics.json` (accuracy, macro/weighted precision/recall/F1, per-class scores, confusion matrix)
- `models/training_history.json` (curves)

Re-evaluate a saved model later with `python -m training.evaluate`. Training on CPU is slow; a GPU or a hosted notebook is recommended for the full dataset. **Report the numbers your own run produces; the app displays nothing else.**

## 11. Model placement

To use a model trained elsewhere (e.g. Colab), copy these into `models/`:

| File | Requirement |
|---|---|
| `skin_model.keras` | Keras model, input `(H, W, 3)` float32 RGB in **[0, 1]** (put any backbone rescaling *inside* the model), softmax output |
| `class_names.json` | JSON list of class slugs in the model's output order, e.g. `["akiec","bcc",...]` |
| `model_metrics.json` | Optional; produced by `training.evaluate`. Without it the analytics page shows an empty state |

The app checks that the number of outputs equals the number of class names and that outputs form a valid probability distribution. Replacing the files while the app runs reloads the model automatically.

## 12. Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///medi_skin.db` | Database location |
| `MAX_UPLOAD_MB` | `5` | Maximum upload size (Streamlit's hard ceiling is set to 25 MB in `.streamlit/config.toml`) |
| `SESSION_TIMEOUT_MINUTES` | `30` | Idle time before automatic sign-out |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MODEL_PATH` | `models/skin_model.keras` | Override model location |
| `CLASS_NAMES_PATH`, `METRICS_PATH`, `HISTORY_PATH` | `models/…` | Override related files |
| `REPORTS_DIR` | `reports/generated` | Where PDFs are written |
| `HOSPITAL_PROVIDER` | `local` | Hospital data provider name |
| `HOSPITAL_API_KEY` | *(empty)* | For a future API provider; never logged or displayed |

Invalid values fall back to safe defaults and a warning is logged. `.env` is git-ignored; commit only `.env.example`. The application needs no secrets to run.

### Hospital data

The app ships with an empty `data/metadata/hospitals.json`. Add real facilities you have verified:

```json
[
  {"name": "Example Skin Clinic", "city": "Your City", "address": "…", "phone": "…",
   "website": "https://…", "dermatology": true, "latitude": 12.34, "longitude": 56.78}
]
```

Only `name` and `city` are required. To connect a live API later, implement `HospitalProvider.search`, call `register_provider("my_api", MyProvider)` and set `HOSPITAL_PROVIDER=my_api`; no page needs to change.

## 13. Running the application

```bash
streamlit run app.py
```

Open http://localhost:8501, create an account, then use **Analyze Skin**. Restart Streamlit after editing Python files (the file watcher does not always reload imported modules).

## 14. Testing

```bash
pytest                           # unit + in-process smoke tests (100 tests)
```

Covers authentication and password hashing, validators and image validation, prediction output and the end-to-end training pipeline (on tiny synthetic data), confidence/risk rules, database operations, report generation, hospitals, feedback, history, analytics, lesion-aware splitting, and an in-process application smoke test (`streamlit.testing`).

Optional real-browser test (Chromium via Playwright):

```bash
pip install playwright && playwright install chromium
python tests/e2e/run_e2e.py                 # registration → upload → analysis → PDF → history → feedback …
python tests/e2e/run_e2e.py --no-model      # the shipped state without a model
```

The browser test builds a throw-away **random-weights** model in a temp folder purely to exercise the UI. It never touches `models/`, the project database or `reports/generated/`, and its predictions are meaningless.

## 15. Screenshots

Add your own after training a model:

| Page | Screenshot |
|---|---|
| Sign in | `docs/screenshots/login.png` *(placeholder)* |
| Dashboard | `docs/screenshots/home.png` *(placeholder)* |
| Analyze | `docs/screenshots/analyze.png` *(placeholder)* |
| Result + Grad-CAM | `docs/screenshots/results.png` *(placeholder)* |
| History | `docs/screenshots/history.png` *(placeholder)* |
| Model analytics | `docs/screenshots/analytics.png` *(placeholder)* |

`python tests/e2e/run_e2e.py --screenshots docs/screenshots` captures every page (with the throw-away test model — retake the result and analytics shots with your real model).

## 16. Limitations

- **No model or dataset is included**; you must train or provide one. Reported performance depends entirely on your data and run.
- Models trained on HAM10000 (dermoscopic images, limited skin-tone diversity, one hospital-sourced collection) may perform poorly on phone photos and on under-represented skin tones. Confident-but-wrong predictions are possible.
- The model only knows its training classes and cannot assess symptoms, history or change over time.
- Triage indicators come from a fixed per-class table plus the confidence band; they are informational, not clinical risk scores.
- Login sessions live in Streamlit session state: refreshing the browser tab signs the user out. The login throttle is in memory (resets on restart, per process).
- SQLite and in-process caching suit a single-instance demo, not a multi-server deployment.
- Hospital results are only as good as the data you add; no live provider is connected.
- Times are stored and shown in UTC.
- Uploaded images are not stored; a PDF report generated by the user embeds the image it was created from (reports made later from history omit it).

## 17. Future enhancements

Live hospital/maps provider and geolocation · persistent "remember me" sessions · role-based admin and feedback dashboard · calibration of model probabilities and out-of-distribution detection · multi-language support · email delivery of reports · model versioning and A/B comparison · containerised deployment with PostgreSQL.

## 18. Medical disclaimer

MediSkin AI is an academic, AI-assisted screening and education tool. It does **not** provide a medical diagnosis, does not recommend treatment or medication, and must not replace examination by a qualified healthcare professional. Do not delay seeking care, or ignore professional advice, because of anything shown by this application. If you notice a new, changing, bleeding or painful skin lesion, see a dermatologist.

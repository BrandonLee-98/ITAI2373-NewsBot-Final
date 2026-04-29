import os

# Base Directory of the Project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Configurations
MODELS = {
    "CLASSIFIER": "valhalla/distilbart-mnli-12-3",
    "SUMMARIZER": "sshleifer/distilbart-cnn-12-6",
    "QA_BOT": "deepset/minilm-uncased-squad2",
    "NER_MODEL": "en_core_web_sm"
}

# Path Configurations [cite: 97-105]
PATHS = {
    "RAW_DATA": os.path.join(BASE_DIR, "data", "raw"),
    "PROCESSED_DATA": os.path.join(BASE_DIR, "data", "processed"),
    "MODEL_STORAGE": os.path.join(BASE_DIR, "models"),
    "REPORTS": os.path.join(BASE_DIR, "reports")
}

# Classification Labels
CANDIDATE_LABELS = ["Finance", "Tech", "Politics", "Health", "Legal"]

# System Settings
APP_DEBUG = True
APP_PORT = 5000
DEFAULT_LANGUAGE = "en"

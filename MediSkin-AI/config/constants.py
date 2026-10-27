"""Static constants: UI text, thresholds, model input size, risk mapping."""

APP_NAME = "MediSkin AI"
APP_TAGLINE = "AI-assisted skin condition screening and healthcare guidance"
APP_VERSION = "1.0.0"

MEDICAL_DISCLAIMER = (
    "This result is for preliminary informational purposes only and is not a medical "
    "diagnosis. Please consult a qualified healthcare professional for proper "
    "evaluation and treatment."
)
GRADCAM_DISCLAIMER = "AI attention visualization — this is not a clinical interpretation."

# --- Image / model -------------------------------------------------------
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
ALLOWED_MIME_FORMATS = {"JPEG", "PNG"}  # real format as detected by Pillow
MODEL_INPUT_SIZE = (224, 224)
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard
TOP_K_PREDICTIONS = 3

# --- Confidence bands (model probability, NOT medical certainty) ---------
HIGH_CONFIDENCE = 0.75
MODERATE_CONFIDENCE = 0.50

CONFIDENCE_HIGH = "High"
CONFIDENCE_MODERATE = "Moderate"
CONFIDENCE_LOW = "Low"

# --- Informational triage categories -------------------------------------
RISK_ROUTINE = "Lower concern"
RISK_MONITOR = "Moderate concern"
RISK_PRIORITY = "Higher concern"
RISK_UNCERTAIN = "Uncertain"

RISK_COLORS = {
    RISK_ROUTINE: "#1B7F5C",
    RISK_MONITOR: "#B45309",
    RISK_PRIORITY: "#B91C1C",
    RISK_UNCERTAIN: "#475569",
}

# Base triage level per class slug. Unknown classes fall back to RISK_MONITOR.
CLASS_BASE_RISK = {
    "nv": RISK_ROUTINE,
    "bkl": RISK_ROUTINE,
    "df": RISK_ROUTINE,
    "vasc": RISK_MONITOR,
    "akiec": RISK_PRIORITY,
    "bcc": RISK_PRIORITY,
    "mel": RISK_PRIORITY,
}

# --- Auth ----------------------------------------------------------------
MIN_PASSWORD_LENGTH = 8
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 254
MAX_COMMENT_LENGTH = 1000

# --- Misc ----------------------------------------------------------------
HISTORY_PAGE_SIZE = 50

# --- General education shown on the dashboard (cautious, non-diagnostic) ----
SKIN_CHECK_GUIDE = (
    ("A", "Asymmetry", "One half of a mole or spot does not match the other half."),
    ("B", "Border", "Edges are irregular, ragged, notched or blurred."),
    ("C", "Colour", "Several shades (browns, black, red, white or blue) in the same spot."),
    ("D", "Diameter", "Larger than about 6 mm (pencil eraser), although smaller spots can matter too."),
    ("E", "Evolving", "Any change in size, shape, colour, or new itching, bleeding or crusting."),
)
EDUCATION_TIPS = (
    ("wb_sunny", "Sun safety",
     "Use broad-spectrum sunscreen, wear protective clothing and seek shade around midday. "
     "Avoid sunburn and tanning beds."),
    ("photo_camera", "Taking a useful photo",
     "Use natural light, keep the spot in sharp focus and fill most of the frame. "
     "Avoid filters, heavy shadows and glare."),
    ("calendar_month", "Check regularly",
     "Look over your whole body every month or two and note anything new, changing, "
     "bleeding or that does not heal."),
    ("medical_services", "See a professional",
     "An app cannot examine your skin. A dermatologist can; see one for anything that worries you."),
)

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM Models
OPENAI_MODEL = "gpt-4.1-nano"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
GEMINI_MODEL = "gemini-1.5-pro"

# Output Directories
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PATIENT_DATA_DIR = os.path.join(OUTPUT_DIR, "patient_data")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Ensure directories exist
for directory in [OUTPUT_DIR, LOG_DIR, PATIENT_DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# CGM Data Parameters
CGM_READING_INTERVAL = 5  # minutes
CGM_READINGS_PER_DAY = 288  # 24 hours * 60 minutes / 5 minutes
CGM_TOTAL_READINGS = 120  # 10 hours of data

# Glucose Thresholds (mg/dL)
HYPOGLYCEMIA_THRESHOLD = 70
HYPERGLYCEMIA_THRESHOLD = 180
TARGET_RANGE_MIN = 70
TARGET_RANGE_MAX = 180

# LLM Providers
LLM_PROVIDERS = ["openai", "anthropic", "gemini"]

# Prompts
PROMPT_TYPES = ["detailed", "concise"]

# Evaluation Criteria
EVALUATION_CRITERIA = [
    "comprehensiveness",
    "specificity",
    "clinical_reasoning",
    "personalization",
    "safety"
]

# SocraSynth Configuration
CONTENTIOUSNESS_LEVELS = [0.8, 0.5, 0.2]


# CRIT Scoring Configuration
CRIT_METHODS = [
    "definition",
    "generalization",
    "induction",
    "elenchus",
    "hypothesis_elimination",
    "maieutic",
    "dialectic"
]

# Weights for calculating overall score (must sum to 1.0)
CRIT_WEIGHTS = {
    "definition": 0.15,
    "generalization": 0.15,
    "induction": 0.20,
    "elenchus": 0.20,
    "hypothesis_elimination": 0.05,
    "maieutic": 0.15,
    "dialectic": 0.10
}

# Default weight to use if a method is missing
CRIT_DEFAULT_WEIGHT = 0.15

# Maximum score value (for normalization)
CRIT_MAX_SCORE = 5

#patient_profile
PATIENT_PROFILE_ONE = {
    "id": "P_001",
    "name": "Alex Johnson",
    "age": 28,
    "gender": "Female",
    "diabetes_type": "Type 1",
    "years_with_diabetes": 15,
    "baseline_glucose": 140,
    "variability": 30,
    "meal_impact": 60,
    "activity_impact": -35,
    "has_dawn_effect": True,
    "meal_times": [8, 13, 19],
    "activity_times": [10, 16],
    "description": "Young adult with clear morning glucose rises (4-8 AM) and moderate overall control with nocturnal hypoglycemia risk.",
    "medications": ["Insulin Glargine (20 units at bedtime)", "Insulin Lispro (4-8 units before meals)"],
    "medical_history": ["Diagnosed with T1D at age 13", "No diabetes complications", "Occasional hypoglycemia unawareness"]
}

PATIENT_PROFILE_FROM_CONFIG = {
    "name": "Alex Johnson",
    "age": 28,
    "gender": "Female",
    "diabetes_type": "Type 1",
    "years_with_diabetes": 15,
    "baseline_glucose": 140,
    "variability": 30,
    "meal_impact": 60,
    "activity_impact": -35,
    "has_dawn_effect": True,
    "meal_times": [8, 13, 19],
    "activity_times": [10, 16],
    "description": "Young adult with clear morning glucose rises (4-8 AM) and moderate overall control with nocturnal hypoglycemia risk.",
    "medications": ["Insulin Glargine (20 units at bedtime)", "Insulin Lispro (4-8 units before meals)"],
    "medical_history": ["Diagnosed with T1D at age 13", "No diabetes complications", "Occasional hypoglycemia unawareness"]
}

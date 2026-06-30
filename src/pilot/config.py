"""Centralized paths + constants for the pilot run."""
from pathlib import Path

# --- Repo / library paths ---
# Resolve the repo root from this file's location (src/pilot/config.py) so the
# project runs wherever it is checked out — laptop disk or the flash drive.
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SRC = REPO_ROOT / "models/aifororcas-livesystem/InferenceSystem/src"
INFERENCE_MODEL_CONFIG = REPO_ROOT / "models/aifororcas-livesystem/InferenceSystem/model/config.yaml"
SCRATCH_DIR = REPO_ROOT / "scratch"
WAV_CHUNKS_DIR = SCRATCH_DIR / "wav_chunks"
LOGS_DIR = REPO_ROOT / "logs"
PILOT_DAY_JSON = LOGS_DIR / "pilot_day.json"

# --- Final library on flash drive ---
LIBRARY_ROOT = Path("/media/y/hlabflash/whale_library")
AUDIO_RAW_DIR = LIBRARY_ROOT / "audio_raw"
AUDIO_CLEAN_DIR = LIBRARY_ROOT / "audio_clean"
SPECTROGRAMS_DIR = LIBRARY_ROOT / "spectrograms"
DB_PATH = LIBRARY_ROOT / "db" / "library.sqlite"
REVIEW_DIR = LIBRARY_ROOT / "review"

# --- Acartia ---
ACARTIA_CSV = Path("/media/y/hlabflash/acartia_2026-02-28.csv")
LAB_LAT, LAB_LON = 48.5583362, -123.1735774  # Orcasound Lab
ACARTIA_RADIUS_KM = 50.0
ACARTIA_TIME_WINDOW_HOURS = 24

# --- Orcasound bucket ---
BUCKET = "audio-orcasound-net"
HYDROPHONE_ID = "rpi_orcasound_lab"

# --- Clip extraction ---
CLIP_DURATION_S = 30.0
DOWNLOAD_CHUNK_S = 60.0  # 60-second WAV chunks downloaded from S3
CLUSTER_GAP_S = 6.0      # merge contiguous positive segments separated by ≤ this many s

# --- Detection ---
HF_REPO_ID = "orcasound/orcahello-srkw-detector-v1"
LOCAL_DETECTION_THRESHOLD = 0.5  # matches model config pred_local_threshold

# --- Multispecies primary detection (humpback scan) ---
MULTISPECIES_SCAN_HOP_S = 2.5      # window hop (s) for the Multispecies scan
MULTISPECIES_DETECT_CLASS = "Mn"   # humpback (Megaptera novaeangliae)
MULTISPECIES_DETECT_THRESHOLD = 0.5

# --- Clip audio format ---
CLIP_SAMPLE_RATE = 48000  # native Orcasound rate per project plan
CLIP_BIT_DEPTH = 16
CLIP_CHANNELS = 1

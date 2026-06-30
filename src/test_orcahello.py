"""Verify OrcaHello loads from HuggingFace and runs on the test WAV.

Imports the InferenceSystem code as a library by adding its src/ to sys.path.
"""
import os
import sys
import time
from pathlib import Path

INFERENCE_SRC = Path(
    "/home/y/whale_acoustic_library/models/aifororcas-livesystem/InferenceSystem/src"
)
TEST_WAV = INFERENCE_SRC.parent / "tests" / "test_data" / (
    "rpi_sunset_bay_2025_09_18_01_12_06_PDT--"
    "f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav"
)

sys.path.insert(0, str(INFERENCE_SRC))

from model.inference import OrcaHelloSRKWDetectorV1  # noqa: E402

REPO_ID = "orcasound/orcahello-srkw-detector-v1"


def main():
    print(f"Loading {REPO_ID} from HuggingFace...")
    t0 = time.time()
    model = OrcaHelloSRKWDetectorV1.from_pretrained(REPO_ID)
    print(f"  loaded in {time.time() - t0:.1f}s")
    p = next(model.parameters())
    print(f"  device: {p.device}, dtype: {p.dtype}")
    model.eval()

    print(f"\nRunning detection on {TEST_WAV.name}")
    t0 = time.time()
    result = model.detect_srkw_from_file(str(TEST_WAV))
    elapsed = time.time() - t0
    print(f"  inference took {elapsed:.1f}s")
    print(f"  result type: {type(result).__name__}")
    # Print key fields without dumping the whole object
    for attr in [
        "global_prediction",
        "global_confidence",
        "local_predictions",
        "segment_predictions",
    ]:
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, list):
                print(f"  {attr}: list of {len(v)} items")
                if v and len(v) <= 5:
                    print(f"    {v}")
                elif v:
                    print(f"    first: {v[0]}")
            else:
                print(f"  {attr}: {v}")


if __name__ == "__main__":
    main()

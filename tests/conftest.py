import os

# Force CPU-friendly + offline-friendly defaults for tests.
os.environ.setdefault("MOCK_GPU", "1")
os.environ.setdefault("LID_OFFLINE_FALLBACK", "1")

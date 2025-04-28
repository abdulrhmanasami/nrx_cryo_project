import json
from pathlib import Path

EXPECTED_VERSION = "1.3.0"
SETTINGS_DIR = Path("settings")

def check_versions():
    if not SETTINGS_DIR.exists() or not SETTINGS_DIR.is_dir():
        print(f"Error: Directory '{SETTINGS_DIR}' not found.")
        return

    for file in SETTINGS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
            print(f"Error reading {file.name}: {str(e)}")
            continue

        version = data.get("_version")
        if version != EXPECTED_VERSION:
            if version is None:
                version_status = "Missing '_version' key"
            else:
                version_status = f"Found {version}"
            print(f"Version mismatch in {file.name}: Expected {EXPECTED_VERSION}, {version_status}")

if __name__ == "__main__":
    check_versions()
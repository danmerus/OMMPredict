import sys
from pathlib import Path

# The app modules live in "Web-application (Flask)/app" and import each other by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Web-application (Flask)" / "app"))

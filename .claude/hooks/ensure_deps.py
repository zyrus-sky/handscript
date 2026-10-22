"""SessionStart hook: make sure the generator's Python packages are installed. Cloud
sessions (claude.ai/code) start on a fresh machine; on this PC it's a quick no-op."""
import importlib.util
import subprocess
import sys

NEED = {"PIL": "Pillow>=10.0", "numpy": "numpy", "cv2": "opencv-python-headless", "fitz": "PyMuPDF",
        "markdown_it": "markdown-it-py", "fontTools": "fonttools", "scipy": "scipy",
        "flask": "flask", "flask_cors": "flask-cors"}
missing = [pkg for mod, pkg in NEED.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)

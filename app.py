"""Hugging Face Spaces entry point for the production OncoLncAI UI."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.main import demo


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("ONCOLNCAI_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("ONCOLNCAI_SERVER_PORT", os.getenv("PORT", "7860"))),
        show_error=True,
    )

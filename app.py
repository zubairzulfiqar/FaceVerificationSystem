"""Entry point.

    python app.py                       # development server
    gunicorn -w 2 -b 0.0.0.0:5000 app:app   # production (used in the Docker image)
"""
import logging
import os

from faceverify.api import create_app

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))

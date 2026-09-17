"""
Root entrypoint for Google Cloud Buildpacks, Gunicorn, and direct execution.
"""
import os
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)


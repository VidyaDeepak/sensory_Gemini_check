FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; pandas wheels cover most needs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run environment defaults
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

COPY main.py .
COPY app ./app
COPY data ./data

# Run gunicorn with Cloud Run best practices:
# - workers 1 + threads 8 for concurrency
# - timeout 0 to let Cloud Run manage request timeouts
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

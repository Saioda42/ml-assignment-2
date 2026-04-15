FROM python:3.11-slim

WORKDIR /app

# Installera system-dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Kopiera requirements och installera dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Kopiera applikation
COPY app/ ./app/
COPY notebooks/ ./notebooks/

# Exponera port
EXPOSE 8000

# Starta servern
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

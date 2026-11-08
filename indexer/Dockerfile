FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --no-create-home appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pipeline ./pipeline
USER 10001
CMD ["python", "-m", "pipeline.run"]

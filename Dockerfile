# --- build stage: compile dlib and every dependency into wheels -----------
FROM python:3.11-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<81" wheel \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- runtime stage: no compilers, non-root user ----------------------------
FROM python:3.11-slim
RUN useradd --create-home app && mkdir /data && chown app /data
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY faceverify faceverify
COPY home home
COPY app.py schema.sql ./
USER app

ENV DB_BACKEND=sqlite \
    SQLITE_PATH=/data/faces.db \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:5000/health').status != 200)"
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "60", "app:app"]

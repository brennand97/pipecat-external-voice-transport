# Pin this tag to an immutable digest before the first production image release.
FROM python:3.12.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir '.[realtime,tools]' \
    && chown -R app:app /app

USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/health', timeout=2)"
ENTRYPOINT ["uvicorn", "voice_transport.app:runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]

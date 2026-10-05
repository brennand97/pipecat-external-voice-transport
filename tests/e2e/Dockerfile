FROM python:3.13.7-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/tests/e2e
WORKDIR /app
COPY tests/e2e/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src /app/src
COPY tests/e2e /app/tests/e2e
USER 1000:1000
EXPOSE 8080
ENTRYPOINT ["uvicorn", "scenario_app:app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]

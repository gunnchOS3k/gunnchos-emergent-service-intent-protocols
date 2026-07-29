FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY schemas ./schemas
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest", "-q"]

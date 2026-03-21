FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy everything first
COPY . .

# Then install — venv ends up at /app/.venv
RUN uv sync --frozen --no-dev

EXPOSE 8000
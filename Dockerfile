FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set up a non-root user for security (Good Practice)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

USER root
# Install uv
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
USER appuser

# Copy only dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Install *only* your *additional* packages
RUN uv sync
# Copy your application code.
COPY --chown=appuser:appuser . .

ENTRYPOINT  ["python","main.py" ]

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to ensure Python output is logged and not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory inside the container
WORKDIR /app

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile into the container
COPY Pipfile ./

# Install dependencies system-wide (since Docker provides isolation, we don't need virtualenvs inside Docker)
RUN pipenv install --system --skip-lock

# Copy the rest of the application code
COPY . .

# Create directories for outputs
RUN mkdir -p data models

# Default command to run the training script
CMD ["python", "src/train.py"]
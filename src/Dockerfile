FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-trained model (produced by train.py in CI/CD)
COPY . .

EXPOSE 9696
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]

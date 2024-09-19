# Use official python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

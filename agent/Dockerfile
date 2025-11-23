FROM python:3.11-slim

WORKDIR /app

# Copy EVERYTHING in the build folder into /app
COPY . /app/

# Run the agent server
CMD ["python", "/app/main.py"]

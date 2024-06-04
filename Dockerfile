# Stage 1: Testing
FROM python:3.10-slim

WORKDIR /usr/src/app
RUN apt update && apt upgrade -y
COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]

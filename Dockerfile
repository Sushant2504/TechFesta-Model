FROM python:3.11.4-slim

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip==25.0 && \
    pip install -r requirements.txt 

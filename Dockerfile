# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /web-b-gone

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --default-timeout=5000

RUN python -c "exec(\"import nltk\nnltk.download('punkt')\nnltk.download('wordnet')\nnltk.download('omw-1.4')\")"
RUN python -m spacy download en_core_web_md

COPY . .

ENTRYPOINT ["python", "startup.py"]
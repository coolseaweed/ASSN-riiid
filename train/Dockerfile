# build base image
FROM python:3.10.11-slim-buster AS base
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
RUN apt-get update && apt-get install -y \
    libpq-dev gcc g++ curl \
    && ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime 


FROM base AS dev
ENV WORKDIR=/workspace
WORKDIR ${WORKDIR}

COPY . ${WORKDIR}
RUN pip install --upgrade pip -r requirements.txt

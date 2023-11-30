# Based on the latest datascience-notebook as of 4/30/21
ARG STARTER_IMAGE=mcr.microsoft.com/devcontainers/universal:2
FROM ${STARTER_IMAGE}

RUN 

# Add in your own requirements below.
# For best maintainability use a specific version.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
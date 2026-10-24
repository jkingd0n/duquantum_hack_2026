FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

USER root

# The base image's unused Yarn repository can have an expired signing key.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglpk40 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==0.13.5

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

USER vscode
WORKDIR /workspaces/duquantum_hack_2026

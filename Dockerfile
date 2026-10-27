# Based on the MLOps HS26 Block 1 exercise, Part C (ship it: multi-stage Dockerfile), F. Bär [5] in README.md.
#
# The feature pipeline as a container image: the same code and the exact package
# versions from uv.lock, on any machine with Docker. The image holds no data and
# no keys: the data is mounted from outside (see "Run").
#
#   builder  python:3.12-slim + uv. The package layer depends only on pyproject.toml
#            and uv.lock, so Docker reuses it as long as the dependencies are the same.
#   final    python:3.12-slim, a non-root user (uid 1001), only the packages and the
#            code - no uv, no pytest, no build tools.
#
# Build:   docker build -t lake-geneva-features .
# Run:     docker run --rm -v "${PWD}/data:/app/data" lake-geneva-features
#          (rebuilds data/features/features.parquet from the raw files in data/raw)
# Other steps of the pipeline, e.g. the water download:
#          docker run --rm -v "${PWD}/data:/app/data" lake-geneva-features python -m lakegeneva.features.ingest_water --out /app/data/raw

# ---- builder: install the exact package versions ---------------------------
# Python 3.12, the same version as .python-version (local, CI and the daily job).
FROM python:3.12-slim AS builder

# uv from its official image, with a fixed version (not "latest"), like my laptop.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/

WORKDIR /app
# Copy only the two dependency files first: as long as they do not change, Docker
# takes the next layer from its cache instead of installing everything again.
COPY pyproject.toml uv.lock ./
# The packages as a plain folder (/app/deps), not a .venv: a .venv contains
# absolute paths and its own Python, so it cannot simply be copied into the next
# stage. --no-dev leaves out pytest; --no-emit-project leaves out our own code,
# which is copied separately below.
RUN uv export --locked --no-dev --no-emit-project -o /tmp/requirements.txt && \
    uv pip install --system --target /app/deps -r /tmp/requirements.txt

# ---- final: small runtime image, non-root -----------------------------------
FROM python:3.12-slim

# Which version of the code is inside: CI passes the Git commit (see ci.yml).
ARG APP_VERSION=dev
# PYTHONPATH: where Python finds the packages and our package "lakegeneva".
ENV APP_VERSION=$APP_VERSION \
    PYTHONPATH=/app/deps:/app/src

# A normal user instead of root: if the code had a security hole, it could not
# change the system inside the container.
RUN groupadd --gid 1001 app && useradd --uid 1001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/deps /app/deps
COPY --chown=app:app src/ /app/src/
# The data folder is mounted at run time; it only has to exist and belong to the user.
RUN mkdir -p /app/data && chown app:app /app/data

USER 1001

# No HEALTHCHECK and no EXPOSE: the feature pipeline is a job that runs and ends,
# not a web service that waits for requests (the web app comes with MS4).
# Exec form (a list): the command runs directly, without a shell in between.
CMD ["python", "-m", "lakegeneva.features.build_features", "--raw", "/app/data/raw", "--out", "/app/data/features/features.parquet"]

# syntax=docker/dockerfile:1
# Minimal CPU image for PanTEon (training + inference + library + evaluation).
# TensorFlow + PyTorch + Transformers dominate image size (~4-6 GB compressed).

# -----------------------------------------------------------------------------
# Stage 1: compile Java utilities and create the Python environment
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/java
COPY features/KmersFeaturesCollector.java features/BufferReaderAndWriter.java ./
RUN javac KmersFeaturesCollector.java BufferReaderAndWriter.java

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && pip install --no-cache-dir --no-deps hierarchicalsoftmax==1.4.4 \
    && pip install --no-cache-dir graphviz==0.21

# -----------------------------------------------------------------------------
# Stage 2: runtime image
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="PanTEon" \
      org.opencontainers.image.description="Deep Learning Framework for Transposable Element Classification (CPU)" \
      org.opencontainers.image.source="https://github.com/simonorozcoarias/PanTEon"

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    ncbi-blast+ \
    perl \
    graphviz \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/PanTEon

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/java/KmersFeaturesCollector.class features/
COPY --from=builder /build/java/BufferReaderAndWriter.class features/

COPY PanTEon.py .
COPY Classifiers/ Classifiers/
COPY Custom_classifiers/ Custom_classifiers/
COPY features/ features/
COPY data/ data/
COPY tools/ tools/
COPY Test_data/ Test_data/

RUN chmod +x features/kanalyze-2.0.0/code/runKanalyzer_generate_all_features \
    && chmod +x tools/RMout_to_bed.pl

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=1

# Recommended runtime volumes:
#   -v /path/to/models:/models
#   -v /path/to/database:/opt/PanTEon/data  (large Zenodo FASTA)
#   -v /path/to/output:/work
ENTRYPOINT ["python", "PanTEon.py"]

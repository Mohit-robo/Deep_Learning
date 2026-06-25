# ─────────────────────────────────────────────────────────
# OneEye — DeepStream Backend Dockerfile
# Base: NVIDIA DeepStream 7.1 with Triton (multi-arch)
# ─────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# ── Configurable ports ───────────────────────────────────
ENV GRPC_PORT=50051
ENV RTSP_PORT=8554

# ── System deps ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-tools \
    wget \
    curl \
    net-tools \
    iputils-ping \
    libflac8 \
    libmpg123-0 \
    libmp3lame0 \
    mjpegtools \
    ffmpeg \
    libopencv-dev --allow-unauthenticated \
    && rm -rf /var/lib/apt/lists/*

# ── Install DeepStream Python bindings (pyds) ───────────
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl && \
    python3 -m pip install --no-cache-dir ./pyds*.whl && \
    rm ./pyds*.whl

# ── Expose ports ─────────────────────────────────────────
# gRPC control API
EXPOSE 50051

# RTSP server port
EXPOSE 8554

# Optional RTP/RTCP UDP ports for smoother local streaming
# (useful if RTSP client negotiates UDP transport)
EXPOSE 5400-5500/udp

# ── Deploy custom library overrides ──────────────────────
# See dockerfiles/lib-container-paths.md for full mapping.
ENV DS_ROOT=/opt/nvidia/deepstream/deepstream-7.1

# Runtime libraries (take effect immediately, no recompile needed)
COPY libs/libnvds_msgconv.so               $DS_ROOT/lib/libnvds_msgconv.so
COPY libs/libnvdsgst_nvmultiurisrcbin.so   $DS_ROOT/lib/gst-plugins/libnvdsgst_nvmultiurisrcbin.so
COPY libs/libnvdsgst_nvurisrcbin.so        $DS_ROOT/lib/gst-plugins/libnvdsgst_nvurisrcbin.so

# Custom headers
COPY libs/nvdsmeta_schema.h                $DS_ROOT/sources/includes/nvdsmeta_schema.h
COPY libs/gst-nvdscommonconfig.h           $DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gst-nvdscommonconfig.h
COPY libs/gstdsnvurisrcbin.h               $DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.h

# Custom sources (for reference / recompile inside container)
COPY libs/deepstream_source_bin.c          $DS_ROOT/sources/apps/apps-common/src/deepstream_source_bin.c
COPY libs/eventmsg_payload.cpp             $DS_ROOT/sources/libs/nvmsgconv/eventmsg_payload.cpp
COPY libs/gstdsnvmultiurisrcbin.cpp        $DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp
COPY libs/gstdsnvurisrcbin.cpp             $DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.cpp

# ── Working directory ────────────────────────────────────
WORKDIR /app


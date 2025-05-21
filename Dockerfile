# Base image with CUDA support
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ca-certificates wget tar xz-utils git curl \
    fonts-liberation fontconfig build-essential \
    yasm cmake ninja-build nasm \
    libssl-dev libvpx-dev libx264-dev libx265-dev libnuma-dev libass-dev \
    libmp3lame-dev libopus-dev libvorbis-dev libtheora-dev libspeex-dev \
    libfreetype6-dev libfontconfig1-dev libgnutls28-dev \
    libzimg-dev libwebp-dev pkg-config autoconf automake libtool \
    libfribidi-dev libharfbuzz-dev \
    && rm -rf /var/lib/apt/lists/*

# Point python/pip to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install fdk-aac
RUN git clone https://github.com/mstorsjo/fdk-aac.git && \
    cd fdk-aac && autoreconf -fiv && ./configure && \
    make -j$(nproc) && make install && cd .. && rm -rf fdk-aac

# Install NVENC headers
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && make -j$(nproc) && make install && cd .. && rm -rf nv-codec-headers

# Build and install FFmpeg with NVENC & CUDA support
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && git checkout n7.0.2 && \
    ./configure --prefix=/usr/local \
        --enable-gpl \
        --enable-cuda --enable-cuvid --enable-nvenc \
        --extra-cflags="-I/usr/local/cuda/include -I/usr/include/freetype2" \
        --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu" \
        --enable-pthreads \
        --enable-libx264 --enable-libx265 --enable-libvpx \
        --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libtheora \
        --enable-libspeex --enable-libfreetype --enable-libfribidi --enable-libharfbuzz \
        --enable-libwebp --enable-libfontconfig --enable-libfdk_aac \
        --enable-nonfree --enable-libass\
    && make -j$(nproc) && make install && cd .. && rm -rf ffmpeg

# Add FFmpeg to path
ENV PATH="/usr/local/bin:${PATH}"

# Copy fonts
COPY ./fonts /usr/share/fonts/custom
RUN fc-cache -f -v

# Set app working directory
WORKDIR /app

# Whisper model cache path
ENV WHISPER_CACHE_DIR="/app/whisper_cache"
RUN mkdir -p ${WHISPER_CACHE_DIR}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install openai-whisper jsonschema

# Create non-root app user
RUN useradd -m appuser && chown appuser:appuser /app
USER appuser

# Preload Whisper model
RUN python -c "import os; print(os.environ.get('WHISPER_CACHE_DIR')); import whisper; whisper.load_model('base')"

# Copy the rest of your code
COPY . .

# Expose app port
EXPOSE 8080
ENV PYTHONUNBUFFERED=1

# Run the main app
CMD ["python", "app.py"]
# Pacman ML (Pygame) - run with display forwarding on Linux, or use Dockerfile.web for browser.
#
# Build:  docker build -t pacman-ml .
# Linux (see game window on host):
#   xhost +local:docker
#   docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm pacman-ml
# macOS: use Dockerfile.web (HTML5) or run the app locally; Pygame in Docker needs XQuartz + socat.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents environments unsupervised ml training main.py ./

CMD ["python3", "main.py"]

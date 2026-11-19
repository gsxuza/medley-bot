FROM python:3.12-slim-bookworm

# ffmpeg do Debian já vem com o filtro rubberband (mudança de tom com boa qualidade)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY medley ./medley
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
 && useradd --create-home --uid 1000 bot \
 && mkdir -p /app/data && chown -R bot /app/data \
 && chown -R bot /usr/local/lib/python3.12/site-packages /usr/local/bin

USER bot
ENV DATA_DIR=/app/data PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba
VOLUME /app/data
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "medley.bot"]

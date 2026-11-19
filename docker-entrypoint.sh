#!/bin/sh
# O YouTube muda com frequência e quebra versões antigas do yt-dlp.
# Atualiza a cada inicialização (desligue com YTDLP_AUTO_UPDATE=0).
if [ "${YTDLP_AUTO_UPDATE:-1}" = "1" ]; then
  pip install --no-cache-dir -q -U "yt-dlp[default]" 2>&1 | grep -v -i "warning" || true
fi
exec "$@"

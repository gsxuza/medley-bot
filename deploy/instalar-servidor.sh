#!/usr/bin/env bash
# Instala e inicia o Medley Bot num servidor Ubuntu/Debian novo.
# Uso (no servidor):  bash instalar-servidor.sh
set -euo pipefail

REPO="${REPO:-https://github.com/gsxuza/medley-bot}"
DIR="${DIR:-$HOME/medley-bot}"

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Instalando o Docker"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi
SUDO=""
docker info >/dev/null 2>&1 || SUDO="sudo"

if [ ! -d "$DIR/.git" ]; then
  echo "==> Baixando o projeto em $DIR"
  git clone "$REPO" "$DIR"
fi
cd "$DIR"
mkdir -p data
# O container roda como o usuário 1000
$SUDO chown -R 1000:1000 data

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "==> Criei o arquivo $DIR/.env"
  echo "    Preencha ANTHROPIC_API_KEY e TELEGRAM_BOT_TOKEN:  nano $DIR/.env"
  echo "    Depois rode de novo:  bash deploy/instalar-servidor.sh"
  exit 0
fi

echo "==> Construindo e iniciando o bot"
$SUDO docker compose up -d --build
echo
echo "Pronto! Para ver os logs:  cd $DIR && $SUDO docker compose logs -f"

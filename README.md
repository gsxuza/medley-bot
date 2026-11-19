# Medley Bot 💃🎶

Bot de Telegram que junta e corta músicas para coreografias. Você manda links do YouTube
(ou arquivos de áudio), explica o que quer e recebe o MP3 pronto, com transições na batida e
volume igualado entre as músicas.

## Como funciona

```
Mensagem no Telegram ──► baixa e analisa (yt-dlp + librosa: BPM, batidas, tom, energia)
                     ──► a IA (Gemini ou Claude) transforma o pedido num PLANO (JSON)
                     ──► motor de áudio executa o plano (numpy + ffmpeg)
                     ──► devolve o MP3 + a linha do tempo
```

A IA nunca mexe no áudio, ela só escreve e edita o plano. Para ajustar, ela pede "aumenta a
transição", "começa 5 segundos antes" etc., e o bot gera uma nova versão.

## Instalar (Windows, Mac ou Linux)

1. Instale o **Python 3.11+** e o **ffmpeg**:
   - Windows: `winget install Python.Python.3.12` e `winget install Gyan.FFmpeg`
   - Mac: `brew install python ffmpeg`
   - Linux: `sudo apt install python3-venv ffmpeg`
2. Na pasta do projeto:
   ```bash
   python -m venv .venv
   ```
   Ative o ambiente (Windows: `.venv\Scripts\activate`; Mac/Linux: `source .venv/bin/activate`) e instale:
   ```bash
   pip install -r requirements.txt
   ```
3. Copie `.env.example` para `.env` e preencha:
   - **IA (escolha uma):**
     - Gemini, gratuito: crie a chave em https://aistudio.google.com/apikey e coloque em
       `GEMINI_API_KEY`, com `LLM_PROVIDER=gemini`. O plano gratuito tem limite de pedidos por
       minuto e por dia (de sobra para uso pessoal), e o Google pode usar as conversas do plano
       gratuito para melhorar os produtos dele.
     - Claude, pago por uso: crie a chave em https://console.anthropic.com e coloque em
       `ANTHROPIC_API_KEY`, com `LLM_PROVIDER=anthropic`.
   - `TELEGRAM_BOT_TOKEN`: no Telegram, fale com **@BotFather**, mande `/newbot` e siga os passos.

## Usar

**Bot do Telegram:**
```bash
python -m medley.bot
```
Mande `/start` para o bot. Ele mostra o seu ID; coloque os IDs (o seu e o dela) em
`ALLOWED_USER_IDS` no `.env` e reinicie o bot, para ninguém mais gastar seus créditos.

**Conversa pelo terminal (sem Telegram):**
```bash
python -m medley.cli chat
```

**Só o motor de áudio (sem IA), com um plano escrito à mão:**
```bash
python -m medley.cli render plano.json musica1.mp3 musica2.mp3 -o medley.mp3
```

## O plano

```json
{
  "segments": [
    {"track": "T1", "start": 5, "end": 95, "transition": "crossfade", "transition_seconds": 3,
     "gap_before": 0, "semitones": 0, "volume_db": 0},
    {"track": "T2", "start": 62, "end": 180, "transition": "crossfade", "transition_seconds": 3,
     "gap_before": 0, "semitones": 1, "volume_db": 0}
  ],
  "fade_in": 0, "fade_out": 3, "snap": "phrase", "speed": 1.0
}
```

- `start`/`end`: segundos na música original. O crossfade acontece nos últimos N segundos de
  um trecho e nos primeiros N do seguinte.
- `transition`: `crossfade` (sobrepõe), `cut` (corte seco) ou `fade` (abaixa, depois entra a próxima).
- `snap`: `phrase` ajusta cada corte para o começo da contagem de 8 mais próxima (o "1" da dança);
  `beat`, para a batida mais próxima; `none` usa o tempo exato.
- `semitones`: muda o tom. Usa o filtro rubberband do ffmpeg se existir; se não, o librosa (qualidade menor).
- `speed`: 0.85 = 85% da velocidade, sem mudar o tom (bom para ensaio).

## Rodar no servidor

O bot usa *polling* (ele busca as mensagens no Telegram), então o servidor **não precisa de
domínio, HTTPS nem porta aberta**. Basta uma máquina Linux ligada com internet.

**Servidor recomendado:** VPS com Ubuntu 24.04, **2 GB de RAM** e 20 GB de disco (o processamento
carrega as músicas na memória). Qualquer provedor serve: Hetzner, DigitalOcean, Hostinger,
Magalu Cloud, Oracle Cloud (tem plano gratuito) etc.

### Instalação

No servidor (via SSH), clone o projeto. Como o repositório é privado, o `git clone` vai pedir
seu usuário do GitHub e, no lugar da senha, um token criado em
https://github.com/settings/tokens (com acesso de leitura ao repositório):

```bash
git clone https://github.com/gsxuza/medley-bot && cd medley-bot
```

```bash
bash deploy/instalar-servidor.sh
```

Na primeira vez o script instala o Docker e cria o `.env`. Preencha as chaves
(`nano .env`) e rode o script de novo para subir o bot.

### Dia a dia

| Para... | Comando (na pasta do projeto) |
|---|---|
| ver os logs | `docker compose logs -f` |
| reiniciar | `docker compose restart` |
| parar | `docker compose down` |
| atualizar o código | `git pull && docker compose up -d --build` |

- O bot volta sozinho se o servidor reiniciar (`restart: unless-stopped`).
- O `yt-dlp` é atualizado toda vez que o container inicia, porque o YouTube muda com frequência.
  Se downloads começarem a falhar, um `docker compose restart` costuma resolver.
- Conversas, áudios e medleys ficam em `data/`. Arquivos sem uso há mais de `KEEP_DAYS` dias
  (padrão 30) são apagados; se ela voltar numa conversa antiga, o bot baixa as músicas de novo.

### Se o YouTube bloquear o servidor

O YouTube às vezes bloqueia IPs de datacenter ("Sign in to confirm you're not a bot"). Opções,
da mais simples para a mais trabalhosa:

1. Ela manda o arquivo de áudio direto no chat (sempre funciona).
2. Cookies: no seu navegador, com uma conta Google **secundária** logada no YouTube, exporte os
   cookies com a extensão "Get cookies.txt LOCALLY", salve como `data/cookies.txt` no servidor e
   coloque `YTDLP_COOKIES=/app/data/cookies.txt` no `.env`. Não use sua conta principal, porque
   o Google pode bloquear contas usadas assim.
3. Rodar o bot em casa (um PC antigo ou mini PC), que usa IP residencial.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `medley/audio.py` | download (yt-dlp), conversão e análise |
| `medley/render.py` | executa o plano e gera o MP3 |
| `medley/plan.py` | formato do plano (também é o schema que a IA preenche) |
| `medley/agent.py` | prompt e chamada à IA (Gemini ou Claude) |
| `medley/session.py` | estado de cada conversa (salvo em `data/chats/`) |
| `medley/core.py` | fluxo de uma mensagem |
| `medley/bot.py` / `medley/cli.py` | Telegram / terminal |
| `Dockerfile`, `docker-compose.yml` | imagem e execução no servidor |
| `deploy/instalar-servidor.sh` | instalação num servidor Ubuntu/Debian |

## O que a IA sabe sobre cada música

A IA não ouve o áudio. Antes de cada pedido ela recebe, de cada música:

- duração, andamento (BPM) e tom estimado;
- onde começa cada contagem de 8 (detectada pelas trocas de acorde, bumbo e pratos);
- a **letra com horários**, tirada da legenda do YouTube (manual ou automática), com as frases
  repetidas marcadas como provável refrão;
- os capítulos do vídeo, quando existem, e uma curva de energia (partes mais fortes ou mais fracas).

Com isso, pedidos como "começa no refrão" ou "corta depois de 'ressuscita-me'" caem no lugar
certo, e a linha do tempo mostra a frase com que cada trecho começa.

## Limitações conhecidas

- Nem todo vídeo tem legenda. Sem ela, o bot estima as partes da música pela curva de energia,
  que é menos preciso. Legendas automáticas também podem errar palavras ou estar um pouco
  adiantadas.
- A contagem de 8 é estimada. Em músicas sem bateria marcada (só voz e piano, por exemplo)
  ela pode cair na metade errada da contagem.
- Crossfade entre músicas com BPMs muito diferentes pode embolar. Nesses casos o bot prefere
  transições curtas ou fade.
- O YouTube às vezes bloqueia downloads (principalmente de servidores na nuvem). Veja
  "Se o YouTube bloquear o servidor" acima.
- Bots do Telegram só baixam arquivos de até 20 MB.

"""Bot do Telegram. Rode com:  python -m medley.bot"""
import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .audio import cleanup, has_rubberband
from .core import handle
from .session import Session

load_dotenv()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("medley")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
client = anthropic.Anthropic()
locks: dict[int, asyncio.Lock] = {}

HELP = (
    "Oi! Eu monto medleys e cortes de música pra dança 💃\n\n"
    "Me manda os links do YouTube (ou os arquivos de áudio) e explica como quer. Exemplos:\n"
    "• \"Junta essas duas, a segunda começando no refrão\"\n"
    "• \"Corta a primeira de 0:15 até 2:30\"\n"
    "• \"Deixa a transição mais longa\" / \"tira o silêncio do começo\"\n"
    "• \"Me manda uma versão 85% pra ensaiar\"\n\n"
    "/novo começa um medley do zero."
)


def _session(chat_id: int) -> Session:
    return Session.load(DATA_DIR / "chats" / f"{chat_id}.json")


def _allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP + f"\n\n(seu ID do Telegram: {update.effective_user.id})")


async def novo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return
    _session(update.effective_chat.id).reset()
    await update.message.reply_text("Pronto, começamos do zero. Manda as músicas! 🎶")


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await update.message.reply_text("Desculpe, este bot é privado.")
        return
    msg = update.message
    chat_id = update.effective_chat.id
    lock = locks.setdefault(chat_id, asyncio.Lock())
    loop = asyncio.get_running_loop()

    async with lock:
        status = await msg.reply_text("Recebi! Um momento…")
        await ctx.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VOICE)

        def progress(text: str) -> None:
            asyncio.run_coroutine_threadsafe(status.edit_text(text), loop)

        with tempfile.TemporaryDirectory() as tmp:
            files = []
            media = msg.audio or msg.voice or msg.document
            if media:
                f = await media.get_file()
                name = getattr(media, "file_name", None) or "audio"
                path = Path(tmp) / name
                await f.download_to_drive(path)
                title = getattr(media, "title", None) or Path(name).stem
                files.append((path, title))

            text = msg.text or msg.caption or ""
            session = _session(chat_id)
            try:
                result = await asyncio.to_thread(handle, client, session, text, DATA_DIR, files, progress)
            except Exception:
                log.exception("erro processando mensagem")
                await status.edit_text("Ops, algo deu errado do meu lado. Tenta de novo?")
                return

        await status.delete()
        await msg.reply_text(result.reply)
        if result.mp3:
            with open(result.mp3, "rb") as f:
                await msg.reply_audio(
                    f, filename=result.mp3.name, title=f"Medley v{session.version}",
                    caption=result.timeline[:1000],
                )


def _startup_checks() -> None:
    for tool in ("ffmpeg",):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} não encontrado. Instale o ffmpeg.")
    if not shutil.which("deno"):
        log.warning("deno não encontrado: o yt-dlp pode falhar em alguns vídeos do YouTube.")
    if not ALLOWED:
        log.warning("ALLOWED_USER_IDS vazio: qualquer pessoa que achar o bot pode usá-lo.")
    if not has_rubberband():
        log.warning("ffmpeg sem rubberband: mudança de tom vai usar o librosa (qualidade menor).")


def _cleanup_loop() -> None:
    keep_days = float(os.getenv("KEEP_DAYS", "30"))
    while True:
        try:
            n = cleanup(DATA_DIR, keep_days)
            if n:
                log.info("limpeza: %d arquivos antigos apagados", n)
        except Exception:
            log.exception("erro na limpeza")
        time.sleep(6 * 3600)


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    _startup_checks()
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "ajuda"], start))
    app.add_handler(CommandHandler("novo", novo))
    app.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) | filters.AUDIO | filters.VOICE | filters.Document.AUDIO,
        on_message,
    ))
    log.info("Bot rodando. Ctrl+C para parar.")
    app.run_polling()


if __name__ == "__main__":
    main()

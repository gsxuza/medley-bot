"""Letra com horários, a partir das legendas do YouTube (manuais ou automáticas)."""
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .audio import fmt_time

log = logging.getLogger("medley")

MAX_LINES = 160
_NOISE = re.compile(r"^[\[(].*[\])]$")  # [Música], (Aplausos), [Music]...


def pick_language(info: dict) -> tuple[str, bool] | None:
    """Escolhe a legenda mais útil: manual em português > automática no idioma original > automática em pt."""
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    for k in manual:
        if k.lower().startswith("pt"):
            return k, False
    for k in auto:
        if k.endswith("-orig"):  # legenda automática no idioma em que a música é cantada
            return k, True
    for k in ("pt-BR", "pt"):
        if k in auto:
            return k, True
    for k in manual:  # música em outro idioma com legenda manual
        return k, False
    return None


def parse_json3(text: str) -> list[dict]:
    lines = []
    for ev in json.loads(text).get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        t = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if t:
            lines.append({"t": round(ev.get("tStartMs", 0) / 1000, 2), "text": t})
    return lines


_VTT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->")


def parse_vtt(text: str) -> list[dict]:
    lines, start = [], None
    for raw in text.splitlines():
        m = _VTT_TIME.match(raw.strip())
        if m:
            h, mi, s, ms = map(int, m.groups())
            start = h * 3600 + mi * 60 + s + ms / 1000
            continue
        txt = re.sub(r"<[^>]+>", "", raw).strip()
        if start is not None and txt and not txt.isdigit():
            lines.append({"t": round(start, 2), "text": txt})
    return lines


def clean(lines: list[dict]) -> list[dict]:
    out = []
    for ln in sorted(lines, key=lambda x: x["t"]):
        text = re.sub(r"[♪♫]", "", ln["text"]).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or _NOISE.match(text):
            continue
        # Legendas automáticas repetem a linha anterior logo em seguida; repetição de verdade (refrão
        # cantado duas vezes) vem alguns segundos depois e deve ficar.
        if out and out[-1]["text"] == text and ln["t"] - out[-1]["t"] < 2.0:
            continue
        out.append({"t": ln["t"], "text": text})
    return out


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def fetch(url: str, info: dict, cache_dir: Path, base_opts: dict) -> tuple[list[dict], bool]:
    """Baixa e lê a legenda. Retorna (linhas, verificado). verificado=False se deu erro e vale tentar de novo."""
    vid = info["id"]
    cached = cache_dir / f"{vid}.lyrics.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8")), True

    choice = pick_language(info)
    lines: list[dict] = []
    if choice:
        lang, is_auto = choice
        import yt_dlp

        opts = {
            **base_opts,
            "skip_download": True,
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "json3/vtt/best",
            "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log.warning("não consegui baixar a legenda de %s: %s", url, e)
            return [], False
        for f in cache_dir.glob(f"{vid}.{lang}.*"):
            try:
                text = f.read_text(encoding="utf-8")
                lines = parse_json3(text) if f.suffix == ".json3" else parse_vtt(text)
            except Exception as e:
                log.warning("legenda ilegível %s: %s", f.name, e)
            f.unlink(missing_ok=True)
        lines = clean(lines)
        log.info("legenda %s (%s): %d linhas", lang, "automática" if is_auto else "manual", len(lines))

    cached.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    return lines, True


def describe(lines: list[dict]) -> str:
    """Texto para a IA: cada linha com horário; ↻ marca frases que se repetem (provável refrão)."""
    counts = Counter(_norm(ln["text"]) for ln in lines)
    shown = lines[:MAX_LINES]
    out = []
    for ln in shown:
        mark = " ↻" if counts[_norm(ln["text"])] >= 2 else ""
        out.append(f"    [{fmt_time(ln['t'])}] {ln['text']}{mark}")
    if len(lines) > len(shown):
        out.append(f"    (… mais {len(lines) - len(shown)} linhas)")
    return "\n".join(out)


def line_near(lines: list[dict], t: float, window: float = 8.0) -> str | None:
    """Primeira frase cantada a partir de t (para mostrar onde cada trecho começa)."""
    for ln in lines:
        if t - 0.5 <= ln["t"] <= t + window:
            return ln["text"]
    return None

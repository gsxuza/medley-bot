"""Download, conversão e análise das músicas."""
import functools
import hashlib
import json
import os
import shutil
import time
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100

# Perfis de Krumhansl-Schmuckler para estimar a tonalidade
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class TrackInfo:
    id: str            # "T1", "T2"... (dado pela conversa)
    source: str        # URL ou nome do arquivo
    title: str
    wav: str           # caminho do WAV convertido
    duration: float
    bpm: float = 0.0
    key: str = ""
    beats: list[float] = field(default_factory=list)
    energy: list[int] = field(default_factory=list)   # 0-9 a cada ENERGY_STEP segundos
    chapters: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Texto compacto que o agente recebe sobre a faixa."""
        lines = [
            f"{self.id}: \"{self.title}\" | duração {fmt_time(self.duration)} ({self.duration:.1f}s)"
            f" | ~{self.bpm:.0f} BPM | tom estimado {self.key}"
        ]
        if self.chapters:
            ch = ", ".join(f"{fmt_time(c['start'])} {c['title']}" for c in self.chapters)
            lines.append(f"  capítulos do YouTube: {ch}")
        if self.energy:
            lines.append(
                f"  energia (0-9) a cada {ENERGY_STEP}s, começando em 0:00: "
                + "".join(str(e) for e in self.energy)
            )
        return "\n".join(lines)


ENERGY_STEP = 5


@functools.lru_cache(maxsize=1)
def ffmpeg() -> str:
    """Caminho do ffmpeg: o instalado no sistema ou, se não houver, o que vem no pacote imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg não encontrado. Rode: pip install -r requirements.txt (ou instale o ffmpeg)"
        ) from None


def fmt_time(t: float) -> str:
    t = max(0, t)
    return f"{int(t // 60)}:{t % 60:04.1f}" if t % 1 else f"{int(t // 60)}:{int(t % 60):02d}"


def _to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vn",
         "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def download_youtube(url: str, cache_dir: Path) -> tuple[Path, dict]:
    """Baixa só o áudio de um link do YouTube. Retorna (wav, metadados)."""
    import yt_dlp

    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    # Em servidores, o YouTube às vezes exige login ("confirme que você não é um robô").
    # Exporte os cookies do navegador para um arquivo e aponte YTDLP_COOKIES para ele.
    cookies = os.getenv("YTDLP_COOKIES")
    if cookies and Path(cookies).is_file():
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        wav = cache_dir / f"{info['id']}.wav"
        if not wav.exists():
            info = ydl.extract_info(url, download=True)
            raw = Path(ydl.prepare_filename(info))
            _to_wav(raw, wav)
            raw.unlink(missing_ok=True)
    meta = {
        "title": info.get("title") or url,
        "chapters": [
            {"start": float(c["start_time"]), "title": c.get("title", "")}
            for c in (info.get("chapters") or [])
        ],
    }
    return wav, meta


def import_file(path: Path, cache_dir: Path, title: str | None = None) -> tuple[Path, dict]:
    """Converte um arquivo de áudio local (mp3, m4a, ogg...) para WAV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    wav = cache_dir / f"file_{digest}.wav"
    if not wav.exists():
        _to_wav(path, wav)
    return wav, {"title": title or path.stem, "chapters": []}


def load(wav: str | Path) -> np.ndarray:
    """Retorna áudio float32 com shape (amostras, 2)."""
    os.utime(wav)  # marca como usado, para a limpeza automática não apagar
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{wav}: esperado {SR} Hz, veio {sr}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def analyze(track_id: str, source: str, wav: Path, meta: dict) -> TrackInfo:
    """BPM, batidas, tom e curva de energia. Resultado fica em cache ao lado do WAV."""
    cache = wav.with_suffix(".analysis.json")
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
    else:
        import librosa

        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        best = (-np.inf, "")
        for i in range(12):
            for prof, mode in ((_MAJOR, ""), (_MINOR, "m")):
                c = np.corrcoef(np.roll(prof, i), chroma)[0, 1]
                if c > best[0]:
                    best = (c, _NOTES[i] + mode)

        hop = ENERGY_STEP * sr
        rms = np.array([
            np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y), hop)
        ])
        db = 20 * np.log10(rms)
        top = np.percentile(db, 95)
        energy = np.clip(np.round((db - (top - 20)) / 20 * 9), 0, 9).astype(int)

        d = {
            "duration": len(y) / sr,
            "bpm": float(np.atleast_1d(tempo)[0]),
            "key": best[1],
            "beats": [round(float(b), 3) for b in beats],
            "energy": energy.tolist(),
        }
        cache.write_text(json.dumps(d), encoding="utf-8")

    return TrackInfo(
        id=track_id, source=source, title=meta["title"], wav=str(wav),
        chapters=meta.get("chapters", []), **d,
    )


def track_to_dict(t: TrackInfo) -> dict:
    return asdict(t)


def cleanup(data_dir: Path, keep_days: float) -> int:
    """Apaga áudios baixados e medleys gerados que não são usados há mais de keep_days dias."""
    if keep_days <= 0:
        return 0
    limit = time.time() - keep_days * 86400
    removed = 0
    for sub in ("cache", "out"):
        for f in (data_dir / sub).glob("*"):
            # atime nem sempre é atualizado; mtime + uso recente via touch em load()
            if f.is_file() and f.stat().st_mtime < limit:
                f.unlink(missing_ok=True)
                removed += 1
    return removed


def has_rubberband() -> bool:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " rubberband " in out

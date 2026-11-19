"""Download, conversão e análise das músicas."""
import functools
import hashlib
import json
import os
import shutil
import time
import subprocess
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
log = logging.getLogger("medley")

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
    phrases: list[float] = field(default_factory=list)  # onde começa cada contagem de 8 ("1")
    lyrics: list[dict] = field(default_factory=list)    # [{"t": segundos, "text": frase}]
    lyrics_checked: bool = False                        # já tentou buscar a legenda
    analysis_version: int = 1

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
        if self.phrases:
            lines.append(
                "  contagens de 8 (o '1' da contagem da dança) começam em: "
                + ", ".join(fmt_time(p) for p in self.phrases)
            )
        if self.lyrics:
            from .lyrics import describe

            lines.append("  letra com horários (legenda do YouTube; ↻ = frase que se repete, provável refrão):")
            lines.append(describe(self.lyrics))
        else:
            lines.append("  letra: não disponível para esta faixa")
        return "\n".join(lines)


ENERGY_STEP = 5
ANALYSIS_VERSION = 2  # mude quando a análise mudar, para refazer as análises em cache


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

    from .lyrics import fetch

    try:
        lyrics, checked = fetch(url, info, cache_dir, opts)
    except Exception as e:  # a legenda é um extra: nunca impede o medley
        log.warning("erro lendo a legenda: %s", e)
        lyrics, checked = [], False
    meta = {
        "lyrics": lyrics,
        "lyrics_checked": checked,
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
    d = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if d.get("analysis_version") != ANALYSIS_VERSION:
        import librosa

        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
        # O detector às vezes acha metade/o dobro do andamento; músicas de dança ficam entre 60 e 190 BPM
        while 0 < tempo < 60:
            tempo *= 2
        while tempo > 190:
            tempo /= 2
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, bpm=tempo)
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
            "phrases": [round(float(beats[i]), 3) for i in _phrase_starts(y, sr, beat_frames)],
            "analysis_version": ANALYSIS_VERSION,
        }
        cache.write_text(json.dumps(d), encoding="utf-8")

    return TrackInfo(
        id=track_id, source=source, title=meta["title"], wav=str(wav),
        chapters=meta.get("chapters", []), lyrics=meta.get("lyrics", []),
        lyrics_checked=meta.get("lyrics_checked", True), **d,
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _phrase_starts(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> list[int]:
    """Índices das batidas que são o "1" de cada contagem de 8.

    Ideia: em música, o começo de cada compasso (4 tempos) costuma ter troca de acorde e
    batida mais forte (bumbo, prato). Testa as 4 posições possíveis do compasso e fica com a
    que tem mais trocas de acorde e acentos; depois decide entre as 2 metades da contagem de 8.
    """
    import librosa

    n = len(beat_frames)
    if n < 16:
        return []
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)                       # bumbo / baixo
    high = S[freqs > 5000].sum(axis=0)                     # pratos
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr)  # acordes

    frames = np.append(beat_frames, S.shape[1])
    seg = np.stack([chroma[:, frames[i]:max(frames[i + 1], frames[i] + 1)].mean(axis=1) for i in range(n)])
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
    change = np.r_[0.0, 1 - np.sum(seg[1:] * seg[:-1], axis=1)]  # quanto o acorde mudou nesta batida

    def at(curve: np.ndarray) -> np.ndarray:
        return np.array([curve[max(0, f - 2):f + 3].max() for f in beat_frames])

    accent = _zscore(at(onset)) + _zscore(at(low))
    score = _zscore(change) + 0.5 * accent
    bar = int(np.argmax([score[k::4].mean() for k in range(4)]))
    # Os acordes trocam igual nas duas metades da contagem de 8; o que marca o "1" são os acentos
    # (prato, bumbo mais forte, viradas de bateria terminando).
    half_score = accent + 2 * _zscore(at(high))
    half = max((bar, bar + 4), key=lambda k: half_score[k::8].mean() if k < n else -np.inf)
    return list(range(half % 8, n, 8))


def track_to_dict(t: TrackInfo) -> dict:
    return asdict(t)


def cleanup(data_dir: Path, keep_days: float) -> int:
    """Apaga áudios baixados e medleys gerados que não são usados há mais de keep_days dias."""
    if keep_days <= 0:
        return 0
    limit = time.time() - keep_days * 86400
    removed = 0
    for sub in ("cache", "out"):
        for f in (data_dir / sub).rglob("*"):
            # mtime é atualizado (touch) sempre que o arquivo é usado
            if f.is_file() and f.stat().st_mtime < limit:
                f.unlink(missing_ok=True)
                removed += 1
        for d in sorted((data_dir / sub).rglob("*"), reverse=True):  # pastas que ficaram vazias
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    return removed


def has_rubberband() -> bool:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " rubberband " in out

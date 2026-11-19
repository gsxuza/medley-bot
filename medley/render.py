"""Executa um Plan e gera o MP3 final. Nada aqui usa IA: é determinístico."""
import functools
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, TrackInfo, ffmpeg, fmt_time, has_rubberband, load
from .lyrics import line_near
from .plan import Plan

TARGET_RMS_DB = -16.0   # volume médio que todos os trechos passam a ter
MICRO_FADE = 0.015      # evita estalo nos cortes secos


def _nearest(t: float, points: list[float], max_dist: float) -> float | None:
    if not points:
        return None
    p = np.asarray(points)
    best = float(p[np.argmin(np.abs(p - t))])
    return best if abs(best - t) <= max_dist else None


def _snap(t: float, tr: TrackInfo, mode: str) -> float:
    """Ajusta t para o começo da contagem de 8 (ou batida) mais próxima, se estiver perto."""
    if mode == "none":
        return t
    beat = 60 / tr.bpm if tr.bpm > 0 else 0.5
    if mode == "phrase":
        p = _nearest(t, tr.phrases, max_dist=max(1.5 * beat, 0.8))
        if p is not None:
            return p
    b = _nearest(t, tr.beats, max_dist=0.6)
    return t if b is None else b


def _active_rms_db(x: np.ndarray) -> float:
    mono = x.mean(axis=1)
    win = int(0.4 * SR)
    n = len(mono) // win
    if n == 0:
        return -60.0
    frames = mono[: n * win].reshape(n, win)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    active = db[db > db.max() - 20]       # ignora silêncios e partes muito baixas
    return float(10 * np.log10(np.mean(10 ** (active / 10))))


@functools.lru_cache(maxsize=1)
def _rubberband() -> bool:
    return has_rubberband()


def _pitch_shift(x: np.ndarray, semitones: int) -> np.ndarray:
    if semitones == 0:
        return x
    ratio = 2 ** (semitones / 12)
    if _rubberband():
        return _ffmpeg_filter(x, f"rubberband=pitch={ratio:.6f}:formant=preserved")
    import librosa  # fallback de qualidade menor

    return np.stack(
        [librosa.effects.pitch_shift(x[:, c], sr=SR, n_steps=semitones) for c in range(2)], axis=1
    ).astype(np.float32)


def _ffmpeg_filter(x: np.ndarray, af: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "in.wav", Path(d) / "out.wav"
        sf.write(a, x, SR, subtype="FLOAT")
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(a), "-af", af, str(b)], check=True)
        y, _ = sf.read(b, dtype="float32", always_2d=True)
    return y


def _fade(n: int, kind: str) -> np.ndarray:
    """Curvas de potência constante (o volume não 'afunda' no meio do crossfade)."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    return np.sin(t * np.pi / 2) if kind == "in" else np.cos(t * np.pi / 2)


def render(plan: Plan, tracks: dict[str, TrackInfo], out_mp3: Path) -> list[dict]:
    """Gera o MP3 e retorna a linha do tempo (para mostrar à usuária)."""
    if not plan.segments:
        raise ValueError("O plano não tem nenhum trecho.")

    audio_cache: dict[str, np.ndarray] = {}
    out = np.zeros((0, 2), dtype=np.float32)
    timeline = []

    for i, seg in enumerate(plan.segments):
        tr = tracks.get(seg.track)
        if tr is None:
            raise ValueError(f"Faixa {seg.track} não existe.")
        if seg.track not in audio_cache:
            audio_cache[seg.track] = load(tr.wav)
        src = audio_cache[seg.track]

        start = max(0.0, min(seg.start, tr.duration))
        end = max(start, min(seg.end, tr.duration))
        if start > 0.5:
            start = _snap(start, tr, plan.snap)
        if end < tr.duration - 0.5:
            end = _snap(end, tr, plan.snap)
        if end - start < 0.5:
            raise ValueError(f"Trecho {i + 1} ({seg.track}) ficou curto demais: {start:.1f}s a {end:.1f}s.")

        x = src[int(start * SR): int(end * SR)].copy()
        x = _pitch_shift(x, seg.semitones)
        gain_db = TARGET_RMS_DB - _active_rms_db(x) + seg.volume_db
        x *= np.float32(10 ** (gain_db / 20))

        trans = seg.transition if i > 0 else "cut"
        tsec = max(0.0, seg.transition_seconds)
        gap = int(max(0.0, seg.gap_before) * SR)

        if i == 0:
            n = int(plan.fade_in * SR)
            if n:
                x[:n] *= _fade(min(n, len(x)), "in")[:, None]
            out = x
            pos = 0.0
        elif trans == "crossfade" and gap == 0:
            n = min(int(tsec * SR), len(out), len(x))
            mixed = out[len(out) - n:] * _fade(n, "out")[:, None] + x[:n] * _fade(n, "in")[:, None]
            pos = (len(out) - n) / SR
            out = np.concatenate([out[: len(out) - n], mixed, x[n:]])
        else:
            if trans == "fade":
                n = min(int(tsec * SR), len(out))
                out[len(out) - n:] *= _fade(n, "out")[:, None]
            else:  # corte seco: micro-fade para não estalar
                m = min(int(MICRO_FADE * SR), len(out))
                out[len(out) - m:] *= _fade(m, "out")[:, None]
            m = min(int(MICRO_FADE * SR), len(x))
            x[:m] *= _fade(m, "in")[:, None]
            out = np.concatenate([out, np.zeros((gap, 2), np.float32), x])
            pos = (len(out) - len(x)) / SR

        timeline.append({
            "n": i + 1, "track": seg.track, "title": tr.title,
            "src_start": start, "src_end": end, "out_start": pos,
            "transition": trans if i > 0 else "início",
            "semitones": seg.semitones,
            "lyric": line_near(tr.lyrics, start),
        })

    n = min(int(plan.fade_out * SR), len(out))
    if n:
        out[len(out) - n:] *= _fade(n, "out")[:, None]

    speed = float(np.clip(plan.speed, 0.5, 2.0))
    total = len(out) / SR / speed
    for item in timeline:
        item["out_start"] /= speed

    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "mix.wav"
        sf.write(tmp, out, SR, subtype="FLOAT")
        filters = []
        if abs(speed - 1.0) > 1e-3:
            filters.append(f"atempo={speed:.4f}")
        filters.append("alimiter=limit=0.89:level=disabled")  # segura picos, não muda o volume geral
        subprocess.run(
            [ffmpeg(), "-y", "-loglevel", "error", "-i", str(tmp), "-af", ",".join(filters),
             "-c:a", "libmp3lame", "-b:a", "192k", str(out_mp3)],
            check=True,
        )

    timeline.append({"total": total, "speed": speed})
    return timeline


def describe(timeline: list[dict]) -> str:
    lines = []
    for item in timeline:
        if "total" in item:
            extra = f" (velocidade {item['speed']:.0%})" if abs(item.get("speed", 1) - 1) > 1e-3 else ""
            lines.append(f"Duração total: {fmt_time(item['total'])}{extra}")
            continue
        lines.append(
            f"{fmt_time(item['out_start'])} ▸ {item['track']} \"{item['title']}\" "
            f"({fmt_time(item['src_start'])}–{fmt_time(item['src_end'])}) [{item['transition']}]"
            + (f" [tom {item['semitones']:+d}]" if item.get("semitones") else "")
            + (f"\n    ♪ \"{item['lyric'][:60]}\"" if item.get("lyric") else "")
        )
    return "\n".join(lines)

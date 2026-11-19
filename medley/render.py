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


MATCH_TOLERANCE = 0.08  # só sincroniza o ritmo se os andamentos diferem até 8% (mais que isso soa artificial)
RAMP_BEATS = 8          # depois da mistura, volta ao andamento normal ao longo de 8 batidas
BASS_HZ = 180           # abaixo disso é "grave" (bumbo, baixo)


def _match_ratio(bpm_out: float, bpm_in: float) -> float | None:
    """Quanto acelerar a música que entra para bater com a que sai (considera metade/dobro do andamento)."""
    if bpm_out <= 0 or bpm_in <= 0:
        return None
    best = min((bpm_out * k / bpm_in for k in (0.5, 1.0, 2.0)), key=lambda r: abs(r - 1))
    return best if abs(best - 1) <= MATCH_TOLERANCE else None


def _tempo_ramp(x: np.ndarray, ratio: float, hold: float, ramp: float, semitones: int) -> np.ndarray:
    """Toca x acelerado por `ratio` nos primeiros `hold` segundos e volta aos poucos ao normal em `ramp` segundos.

    Usa o rubberband do ffmpeg (muda andamento sem mudar o tom) com comandos programados no tempo.
    """
    steps = 12
    cmds = [f"{hold + ramp * j / steps:.3f} rubberband tempo {ratio + (1 - ratio) * j / steps:.5f}"
            for j in range(1, steps + 1)]
    pitch = 2 ** (semitones / 12)
    return _ffmpeg_filter(
        x, f"asendcmd=c='{';'.join(cmds)}',rubberband=tempo={ratio:.5f}:pitch={pitch:.6f}:formant=preserved"
    )


@functools.lru_cache(maxsize=1)
def _bass_filter():
    from scipy.signal import butter

    return butter(4, BASS_HZ, btype="low", fs=SR, output="sos")


def _split_bass(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import sosfiltfilt

    low = sosfiltfilt(_bass_filter(), x, axis=0).astype(np.float32)
    return low, x - low


def _dj_mix(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Mistura o fim de uma música com o começo da outra trocando os graves no meio.

    Agudos e médios: um sobe enquanto o outro desce. Graves: só um de cada vez (dois bumbos/baixos
    juntos é o que deixa a transição embolada), trocando rapidamente no meio da mistura.
    """
    n = len(tail)
    t = np.linspace(0, 1, n, dtype=np.float32)
    if n < int(1.5 * SR):  # mistura curta: sem troca de graves
        return tail * _fade(n, "out")[:, None] + head * _fade(n, "in")[:, None]
    lo_a, hi_a = _split_bass(tail)
    lo_b, hi_b = _split_bass(head)
    swap = np.clip((t - 0.5) / 0.25 + 0.5, 0, 1)  # troca dos graves entre 37% e 62% da mistura
    g_in, g_out = np.sin(swap * np.pi / 2), np.cos(swap * np.pi / 2)
    return (hi_a * _fade(n, "out")[:, None] + hi_b * _fade(n, "in")[:, None]
            + lo_a * g_out[:, None] + lo_b * g_in[:, None])


def _echo_tail(x: np.ndarray, bpm: float, repeats: int = 6, feedback: float = 0.55) -> np.ndarray:
    """Eco no ritmo da música: repete a última batida, cada vez mais baixo e sem graves."""
    delay = int(SR * 60 / (bpm if bpm > 0 else 100))
    chunk = x[-delay:] if len(x) >= delay else x
    tail = np.zeros((delay * repeats + len(chunk), 2), np.float32)
    for k in range(1, repeats + 1):
        tail[k * delay - delay: k * delay - delay + len(chunk)] += chunk * feedback ** k
    _, high = _split_bass(tail)
    n = min(len(high), int(0.3 * SR))
    high[-n:] *= _fade(n, "out")[:, None]
    return high


def _beats_in(tr: TrackInfo, start: float, end: float) -> list[float]:
    """Batidas da faixa dentro do trecho, em segundos a partir do início do trecho."""
    return [b - start for b in tr.beats if start - 0.05 <= b <= end]


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
        trans = seg.transition if i > 0 else "cut"
        tsec = max(0.0, seg.transition_seconds)
        gap = int(max(0.0, seg.gap_before) * SR)

        # Mistura no mesmo ritmo: acelera/desacelera a música que entra só durante a mistura
        matched, overlap = False, int(tsec * SR)
        if i > 0 and trans == "crossfade" and gap == 0 and tsec >= 1 and _rubberband():
            p_tr, p_start, p_end = prev
            ratio = _match_ratio(p_tr.bpm, tr.bpm)
            p_beats = _beats_in(p_tr, p_start, p_end)
            in_beats = _beats_in(tr, start, end)
            if ratio and len(p_beats) > 4 and in_beats:
                p_len = p_end - p_start
                # a mistura começa numa batida da música que sai...
                b_k = min((b for b in p_beats if p_len - b >= 1), key=lambda b: abs((p_len - b) - tsec), default=None)
                if b_k is not None:
                    # ...e a primeira batida da que entra cai exatamente nela
                    first = max(0.0, in_beats[0])
                    ov = (p_len - b_k) + first / ratio
                    x = _tempo_ramp(x, ratio, hold=ov * ratio, ramp=RAMP_BEATS * 60 / tr.bpm,
                                    semitones=seg.semitones)
                    matched, overlap = True, int(ov * SR)
        if not matched:
            x = _pitch_shift(x, seg.semitones)

        gain_db = TARGET_RMS_DB - _active_rms_db(x) + seg.volume_db
        x *= np.float32(10 ** (gain_db / 20))

        if i == 0:
            n = int(plan.fade_in * SR)
            if n:
                x[:n] *= _fade(min(n, len(x)), "in")[:, None]
            out = x
            pos = 0.0
        elif trans == "crossfade" and gap == 0:
            n = min(overlap, len(out), len(x))
            mixed = _dj_mix(out[len(out) - n:], x[:n])
            pos = (len(out) - n) / SR
            out = np.concatenate([out[: len(out) - n], mixed, x[n:]])
        else:
            echo = None
            if trans == "echo":
                echo = _echo_tail(out, prev[0].bpm)
            if trans == "fade":
                n = min(int(tsec * SR), len(out))
                out[len(out) - n:] *= _fade(n, "out")[:, None]
            else:  # corte seco (ou eco): micro-fade para não estalar
                m = min(int(MICRO_FADE * SR), len(out))
                out[len(out) - m:] *= _fade(m, "out")[:, None]
            m = min(int(MICRO_FADE * SR), len(x))
            x[:m] *= _fade(m, "in")[:, None]
            cut_at = len(out)
            out = np.concatenate([out, np.zeros((gap, 2), np.float32), x])
            pos = (len(out) - len(x)) / SR
            if echo is not None:  # o eco continua por baixo do silêncio / do começo da próxima
                k = min(len(echo), len(out) - cut_at)
                out[cut_at:cut_at + k] += echo[:k]

        timeline.append({
            "n": i + 1, "track": seg.track, "title": tr.title,
            "src_start": start, "src_end": end, "out_start": pos,
            "transition": trans if i > 0 else "início", "matched": matched, "gap": seg.gap_before,
            "semitones": seg.semitones,
            "lyric": line_near(tr.lyrics, start),
        })
        prev = (tr, start, end)

    if plan.ending == "echo":
        m = min(int(MICRO_FADE * SR), len(out))
        out[len(out) - m:] *= _fade(m, "out")[:, None]
        out = np.concatenate([out, _echo_tail(out, prev[0].bpm)])
    else:
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

    timeline.append({"total": total, "speed": speed, "ending": plan.ending, "fade_out": plan.fade_out})
    return timeline


_LABELS = {
    "início": "início",
    "crossfade": "uma entra enquanto a outra sai",
    "fade": "a anterior abaixa até sumir",
    "cut": "troca direta",
    "echo": "a anterior termina com eco",
}


def describe(timeline: list[dict]) -> str:
    lines = []
    for item in timeline:
        if "total" in item:
            if item.get("ending") == "echo":
                lines.append("Final: termina com eco")
            elif item.get("fade_out", 0) > 0:
                lines.append(f"Final: abaixa até sumir ({item['fade_out']:g} s)")
            extra = f" (velocidade {item['speed']:.0%})" if abs(item.get("speed", 1) - 1) > 1e-3 else ""
            lines.append(f"Duração total: {fmt_time(item['total'])}{extra}")
            continue
        how = _LABELS.get(item["transition"], item["transition"])
        if item.get("matched"):
            how += ", no mesmo ritmo"
        if item.get("gap"):
            how += f", {item['gap']:g} s de silêncio"
        lines.append(
            f"{fmt_time(item['out_start'])} ▸ {item['track']} \"{item['title']}\" "
            f"({fmt_time(item['src_start'])}–{fmt_time(item['src_end'])}) — {how}"
            + (f" — tom {item['semitones']:+d}" if item.get("semitones") else "")
            + (f"\n    ♪ \"{item['lyric'][:60]}\"" if item.get("lyric") else "")
        )
    return "\n".join(lines)

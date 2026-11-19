"""Separa a música em voz, bateria, baixo e outros instrumentos, usando o Demucs (opcional).

Instalação: pip install -r requirements-separacao.txt
Na primeira vez o Demucs baixa o modelo (~80 MB). Separar uma música leva de 1 a alguns minutos
no processador; o resultado fica guardado, então cada música só é separada uma vez.
"""
import importlib.util
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, TrackInfo, load

log = logging.getLogger("medley")

STEMS = ("vocals", "drums", "bass", "other")
# Que partes da música entram em cada opção do plano
MODES: dict[str, tuple[str, ...] | None] = {
    "full": None,                               # música original
    "instrumental": ("drums", "bass", "other"),  # sem a voz
    "vocals": ("vocals",),                       # só a voz
    "drums": ("drums",),                         # só a bateria
    "no_drums": ("vocals", "bass", "other"),     # sem a bateria
}
LABELS = {
    "instrumental": "só instrumental",
    "vocals": "só a voz",
    "drums": "só a bateria",
    "no_drums": "sem bateria",
}

_lock = threading.Lock()  # separar é pesado: uma música por vez
_separator = None


def available() -> bool:
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def stem_dir(wav: str | Path) -> Path:
    wav = Path(wav)
    return wav.with_name(wav.stem + ".stems")


def is_separated(tr: TrackInfo) -> bool:
    d = stem_dir(tr.wav)
    return all((d / f"{s}.flac").exists() for s in STEMS)


def _get_separator():
    global _separator
    if _separator is None:
        from demucs.api import Separator

        log.info("carregando o modelo de separação (na primeira vez ele é baixado, ~80 MB)")
        _separator = Separator(model="htdemucs", progress=False)
    return _separator


def ensure(tr: TrackInfo, progress=lambda msg: None) -> Path:
    """Separa a faixa (se ainda não foi) e retorna a pasta com as partes."""
    d = stem_dir(tr.wav)
    if is_separated(tr):
        return d
    if not available():
        raise RuntimeError("a separação de voz e instrumentos não está instalada neste computador")
    with _lock:
        if is_separated(tr):  # outra conversa pode ter separado enquanto esperávamos
            return d
        import torch

        progress(f"Separando voz e instrumentos de \"{tr.title}\" (leva alguns minutos na primeira vez)…")
        sep = _get_separator()
        last = [0.0]

        def on_progress(info: dict) -> None:
            if info.get("state") != "end" or not info.get("audio_length"):
                return
            done = (info["segment_offset"] / info["audio_length"] + info.get("model_idx_in_bag", 0)) \
                / max(1, info.get("models", 1))
            if time.monotonic() - last[0] > 5:  # não atualiza a mensagem mais que a cada 5 s
                last[0] = time.monotonic()
                progress(f"Separando voz e instrumentos de \"{tr.title}\"… {min(99, int(done * 100))}%")

        sep.update_parameter(callback=on_progress)
        audio = load(tr.wav)
        t0 = time.monotonic()
        with torch.no_grad():
            _, parts = sep.separate_tensor(torch.from_numpy(np.ascontiguousarray(audio.T)), SR)
        d.mkdir(parents=True, exist_ok=True)
        for name in STEMS:
            data = parts[name].cpu().numpy().T
            tmp = d / f"{name}.tmp.flac"
            sf.write(tmp, np.clip(data, -1, 1), sep.samplerate, subtype="PCM_24")
            os.replace(tmp, d / f"{name}.flac")
        log.info("separação de %s levou %.0f s", tr.title, time.monotonic() - t0)
    return d


def load_mix(tr: TrackInfo, mode: str, progress=lambda msg: None) -> np.ndarray:
    """Áudio da faixa só com as partes pedidas (mesmo formato de audio.load)."""
    names = MODES.get(mode)
    if names is None:
        return load(tr.wav)
    d = ensure(tr, progress)
    mix = None
    for name in names:
        f = d / f"{name}.flac"
        os.utime(f)  # marca como usado para a limpeza automática
        data, sr = sf.read(f, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{f}: esperado {SR} Hz, veio {sr}")
        mix = data if mix is None else mix + data
    return mix

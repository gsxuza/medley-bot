"""Estado de cada conversa: faixas, plano atual e histórico. Salvo em JSON."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .audio import TrackInfo, analyze, download_youtube, import_file, track_to_dict
from .plan import Plan

URL_RE = re.compile(r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)/\S+")
MAX_HISTORY = 30  # mensagens (usuária + bot) mantidas como contexto


@dataclass
class Session:
    path: Path
    tracks: dict[str, TrackInfo] = field(default_factory=dict)
    plan: Plan | None = None
    history: list[dict] = field(default_factory=list)
    version: int = 0

    @classmethod
    def load(cls, path: Path) -> "Session":
        s = cls(path=path)
        if path.exists():
            d = json.loads(path.read_text())
            s.tracks = {k: TrackInfo(**v) for k, v in d["tracks"].items()}
            s.plan = Plan.model_validate(d["plan"]) if d.get("plan") else None
            s.history = d.get("history", [])
            s.version = d.get("version", 0)
        return s

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "tracks": {k: track_to_dict(v) for k, v in self.tracks.items()},
            "plan": self.plan.model_dump() if self.plan else None,
            "history": self.history[-MAX_HISTORY:],
            "version": self.version,
        }, ensure_ascii=False))

    def reset(self) -> None:
        self.tracks, self.plan, self.history, self.version = {}, None, [], 0
        self.save()

    def _next_id(self) -> str:
        return f"T{len(self.tracks) + 1}"

    def add_url(self, url: str, cache_dir: Path) -> TrackInfo:
        for t in self.tracks.values():
            if t.source == url:
                return t
        wav, meta = download_youtube(url, cache_dir)
        t = analyze(self._next_id(), url, wav, meta)
        self.tracks[t.id] = t
        return t

    def add_file(self, path: Path, title: str, cache_dir: Path) -> TrackInfo:
        wav, meta = import_file(path, cache_dir, title)
        for t in self.tracks.values():
            if t.wav == str(wav):
                return t
        t = analyze(self._next_id(), title, wav, meta)
        self.tracks[t.id] = t
        return t

    def restore_missing(self, cache_dir: Path) -> list[str]:
        """Rebaixa faixas cujo áudio foi apagado pela limpeza automática."""
        notes = []
        for tid, t in list(self.tracks.items()):
            if Path(t.wav).exists():
                continue
            if URL_RE.fullmatch(t.source):
                wav, meta = download_youtube(t.source, cache_dir)
                self.tracks[tid] = analyze(tid, t.source, wav, meta)
            else:
                del self.tracks[tid]
                notes.append(f"[o áudio da faixa {tid} ({t.title}) expirou; peça para enviar o arquivo de novo]")
        return notes

    def state_text(self) -> str:
        parts = ["FAIXAS DISPONÍVEIS:"]
        parts += [t.summary() for t in self.tracks.values()] or ["(nenhuma ainda)"]
        parts.append("\nPLANO ATUAL:")
        parts.append(self.plan.model_dump_json(indent=1) if self.plan else "(nenhum ainda)")
        return "\n".join(parts)

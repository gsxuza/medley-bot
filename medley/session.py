"""Estado de cada conversa: faixas, plano atual e histórico. Salvo em JSON."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .audio import ANALYSIS_VERSION, TrackInfo, analyze, download_youtube, import_file, track_to_dict
from .plan import Plan

URL_RE = re.compile(r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)/\S+")
MAX_HISTORY = 30  # mensagens (usuária + bot) mantidas como contexto
MAX_PLANS = 20    # versões guardadas para o /desfazer


@dataclass
class Session:
    path: Path
    tracks: dict[str, TrackInfo] = field(default_factory=dict)
    plans: list[Plan] = field(default_factory=list)  # versões do medley, a última é a atual
    history: list[dict] = field(default_factory=list)
    version: int = 0  # número da última versão gerada (só cresce, dá nome aos arquivos)

    @property
    def plan(self) -> Plan | None:
        return self.plans[-1] if self.plans else None

    @property
    def previous_plan(self) -> Plan | None:
        return self.plans[-2] if len(self.plans) >= 2 else None

    def push_plan(self, plan: Plan) -> None:
        self.plans = (self.plans + [plan])[-MAX_PLANS:]
        self.version += 1

    def undo(self) -> Plan | None:
        """Descarta a versão atual. Retorna o plano que voltou a valer (None se não há para onde voltar)."""
        if len(self.plans) < 2:
            return None
        self.plans.pop()
        self.version += 1
        return self.plans[-1]

    @classmethod
    def load(cls, path: Path) -> "Session":
        s = cls(path=path)
        if path.exists():
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:  # arquivos antigos salvos na codificação do Windows
                text = raw.decode("cp1252")
            d = json.loads(text)
            s.tracks = {k: TrackInfo(**v) for k, v in d["tracks"].items()}
            if d.get("plans"):
                s.plans = [Plan.model_validate(p) for p in d["plans"]]
            elif d.get("plan"):  # conversas salvas antes do /desfazer
                s.plans = [Plan.model_validate(d["plan"])]
            s.history = d.get("history", [])
            s.version = d.get("version", 0)
        return s

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "tracks": {k: track_to_dict(v) for k, v in self.tracks.items()},
            "plans": [p.model_dump() for p in self.plans[-MAX_PLANS:]],
            "history": self.history[-MAX_HISTORY:],
            "version": self.version,
        }, ensure_ascii=False), encoding="utf-8")

    def reset(self) -> None:
        self.tracks, self.plans, self.history, self.version = {}, [], [], 0
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

    def refresh_tracks(self, cache_dir: Path) -> list[str]:
        """Rebaixa áudios apagados pela limpeza e atualiza análises antigas (ex.: sem letra/contagem de 8)."""
        notes = []
        for tid, t in list(self.tracks.items()):
            is_url = bool(URL_RE.fullmatch(t.source))
            missing = not Path(t.wav).exists()
            outdated = t.analysis_version < ANALYSIS_VERSION or (is_url and not t.lyrics_checked)
            if not (missing or outdated):
                continue
            if is_url:
                wav, meta = download_youtube(t.source, cache_dir)
                self.tracks[tid] = analyze(tid, t.source, wav, meta)
            elif missing:
                del self.tracks[tid]
                notes.append(f"[o áudio da faixa {tid} ({t.title}) expirou; peça para enviar o arquivo de novo]")
            else:
                meta = {"title": t.title, "chapters": t.chapters, "lyrics": t.lyrics}
                self.tracks[tid] = analyze(tid, t.source, Path(t.wav), meta)
        return notes

    def state_text(self) -> str:
        parts = ["FAIXAS DISPONÍVEIS:"]
        parts += [t.summary() for t in self.tracks.values()] or ["(nenhuma ainda)"]
        parts.append(f"\nPLANO ATUAL (versão {self.version}, a que ela está ouvindo):")
        parts.append(self.plan.model_dump_json(indent=1) if self.plan else "(nenhum ainda)")
        if self.previous_plan:
            parts.append("\nPLANO ANTERIOR (use se ela pedir para voltar como estava antes):")
            parts.append(self.previous_plan.model_dump_json(indent=1))
        return "\n".join(parts)

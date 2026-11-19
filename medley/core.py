"""Fluxo de uma mensagem, usado tanto pelo bot do Telegram quanto pelo modo terminal."""
from dataclasses import dataclass
from pathlib import Path

from . import agent
from .render import describe, render
from .session import URL_RE, Session


@dataclass
class Result:
    reply: str
    mp3: Path | None = None
    timeline: str = ""


def handle(
    session: Session,
    text: str,
    data_dir: Path,
    files: list[tuple[Path, str]] = (),
    progress=lambda msg: None,
) -> Result:
    cache = data_dir / "cache"
    notes = []
    try:
        notes += session.refresh_tracks(cache)
    except Exception as e:
        notes.append(f"[não consegui atualizar uma das músicas desta conversa: {e}]")

    for url in URL_RE.findall(text):
        progress("Baixando e analisando a música…")
        try:
            t = session.add_url(url, cache)
            notes.append(f"[faixa {t.id} adicionada: {t.title}]")
        except Exception as e:  # link inválido, vídeo bloqueado etc.
            notes.append(f"[não consegui baixar {url}: {e}]")

    for path, title in files:
        progress("Analisando o áudio…")
        t = session.add_file(path, title, cache)
        notes.append(f"[faixa {t.id} adicionada: {t.title}]")

    user_text = "\n".join(notes + [text]).strip() or "(sem texto)"

    progress("Pensando na edição…")
    try:
        ans = agent.respond(session, user_text)
    except agent.AgentError as e:
        session.save()
        return Result(f"Tive um problema para entender o pedido ({e}). Pode tentar de novo?")

    result = Result(ans.reply)
    if ans.action == "render" and ans.plan:
        progress("Montando o áudio…")
        try:
            out = data_dir / "out" / f"{session.path.stem}_v{session.version + 1}.mp3"
            timeline = render(ans.plan, session.tracks, out)
        except Exception as e:
            # Deixa o erro no histórico para o agente corrigir na próxima mensagem
            session.history.append({"role": "user", "content": f"[erro ao gerar o áudio: {e}]"})
            result.reply += f"\n\n⚠️ Não consegui gerar o áudio: {e}"
        else:
            session.plan = ans.plan
            session.version += 1
            result.mp3, result.timeline = out, describe(timeline)
    session.save()
    return result

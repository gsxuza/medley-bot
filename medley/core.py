"""Fluxo de uma mensagem, usado tanto pelo bot do Telegram quanto pelo modo terminal."""
from dataclasses import dataclass
from pathlib import Path

import logging

from . import agent
from .render import describe, render
from .session import URL_RE, Session
from .validate import check

log = logging.getLogger("medley")
MAX_FIXES = 2  # quantas vezes a IA pode corrigir o próprio plano


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
        ans, fixes = _decide(session, user_text)
    except agent.AgentError as e:
        session.save()
        return Result(f"Tive um problema para entender o pedido ({e}). Pode tentar de novo?")

    session.history.append({"role": "user", "content": user_text})
    session.history.append({"role": "assistant", "content": ans.reply})
    result = Result(ans.reply)
    if ans.action == "render" and ans.plan:
        progress("Montando o áudio…")
        out = data_dir / "out" / f"{session.path.stem}_v{session.version + 1}.mp3"
        try:
            timeline = render(ans.plan, session.tracks, out, progress)
        except Exception as e:
            log.exception("erro ao gerar o áudio")
            # Deixa o erro no histórico para a IA levar em conta na próxima mensagem
            session.history.append({"role": "user", "content": f"[erro ao gerar o áudio: {e}]"})
            result.reply += f"\n\n⚠️ Não consegui gerar o áudio: {e}"
        else:
            session.push_plan(ans.plan)
            result.mp3, result.timeline = out, describe(timeline)
            if fixes:
                log.info("ajustes automáticos no plano: %s", "; ".join(fixes))
    session.save()
    return result


def _decide(session: Session, user_text: str) -> tuple[agent.AgentResponse, list[str]]:
    """Pede o plano à IA e confere. Se houver problemas, devolve para a IA corrigir (até MAX_FIXES vezes)."""
    followups: list[dict] = []
    for attempt in range(MAX_FIXES + 1):
        ans = agent.respond(session, user_text, followups)
        if ans.action != "render" or not ans.plan:
            return ans, []
        plan, fixes, problems = check(ans.plan, session.tracks)
        if not problems:
            ans.plan = plan
            return ans, fixes
        log.info("plano com problemas (tentativa %d): %s", attempt + 1, "; ".join(problems))
        if attempt == MAX_FIXES:
            break
        followups += [
            {"role": "assistant", "content": ans.model_dump_json()},
            {"role": "user", "content": (
                "[conferência automática — mensagem do sistema, não da usuária] O plano acima tem problemas:\n- "
                + "\n- ".join(problems)
                + "\nCorrija e devolva a resposta completa de novo, com o plano inteiro corrigido. "
                "Não mencione esta conferência na resposta para ela."
            )},
        ]
    lista = "\n".join(f"• {p}" for p in problems)
    return agent.AgentResponse(
        reply=f"Não consegui montar um plano que funcione 😕 Os problemas foram:\n{lista}\n\nPode explicar de outro jeito?",
        action="none", plan=None,
    ), []


def undo(session: Session, data_dir: Path) -> Result:
    """Volta para a versão anterior do medley e gera o áudio dela de novo."""
    plan = session.undo()
    if plan is None:
        return Result("Não tem versão anterior para voltar 🙂")
    out = data_dir / "out" / f"{session.path.stem}_v{session.version}.mp3"
    try:
        timeline = render(plan, session.tracks, out)
    except Exception as e:
        log.exception("erro ao refazer a versão anterior")
        session.save()
        return Result(f"Voltei o plano para a versão anterior, mas não consegui gerar o áudio: {e}")
    msg = "Pronto, voltei para a versão anterior ↩️"
    session.history.append({"role": "user", "content": "[ela usou /desfazer]"})
    session.history.append({"role": "assistant", "content": msg + " (o PLANO ATUAL agora é o que estava antes)"})
    session.save()
    return Result(msg, out, describe(timeline))

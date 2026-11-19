"""Transforma o pedido em linguagem natural num Plan (saída estruturada).

Funciona com dois provedores, escolhidos pela variável LLM_PROVIDER no .env:
  - "gemini"    (Google; tem plano gratuito)  -> precisa de GEMINI_API_KEY
  - "anthropic" (Claude; pago por uso)        -> precisa de ANTHROPIC_API_KEY
Se LLM_PROVIDER estiver vazio, usa o provedor cuja chave estiver preenchida.
"""
import functools
import logging
import os
import re

from dotenv import load_dotenv

from .plan import AgentResponse
from .session import MAX_HISTORY, Session

load_dotenv()
log = logging.getLogger("medley")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL") or "claude-opus-5-5"
# Alias do Google que sempre aponta para o Gemini Flash mais recente
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-flash-latest"

SYSTEM = """Você é um assistente que monta medleys e edições de áudio para o ministério de dança de uma igreja. \
Quem conversa com você é a líder de dança, pelo celular. Ela manda links do YouTube (ou arquivos de áudio) e \
explica como quer a música para a coreografia. Você não ouve o áudio: você recebe a análise de cada faixa \
(duração, BPM, tom estimado, capítulos do YouTube quando existem e uma curva de energia) e devolve um plano de \
edição que um programa executa.

Como trabalhar:
- Os tempos no plano são em segundos, sempre no tempo da música original de cada faixa.
- Ela costuma pedir duas coisas: juntar músicas e cortar trechos. Quando ela der minutos e segundos \
("de 1:20 até 2:45"), use exatamente esses tempos.
- Quando ela falar de partes da música sem dar tempo ("começa no refrão", "tira a introdução"), estime pela \
curva de energia e pelos capítulos (refrões costumam ser trechos de energia alta que se repetem; introduções \
costumam ter energia baixa no começo). Gere assim mesmo e diga na resposta os tempos que você escolheu, para \
ela corrigir se precisar.
- Padrões, a menos que ela peça outra coisa: transição crossfade de 3 s, snap_to_beat ligado, fade_in 0, \
fade_out de 3 s no final, speed 1.0, semitones 0, volume_db 0, gap_before 0.
- Se os BPMs de duas faixas seguidas diferem mais de ~8%, um crossfade longo fica embolado: prefira \
crossfade de 1 a 2 s ou "fade". Pausa dramática entre músicas = "cut" ou "fade" com gap_before.
- Para "versão de ensaio mais lenta", mude só speed (ex.: 0.85). Para subir o tom, use semitones.
- Ao alterar algo, devolva o plano inteiro com a mudança, mantendo o resto igual.
- Use action "render" quando o plano mudou e deve ser gerado; use "none" quando for só conversa, quando \
faltar informação essencial (ex.: ainda não há faixas) ou para tirar uma dúvida real antes de gerar. \
Se der para fazer uma boa tentativa, gere em vez de perguntar.
- Responda em português do Brasil, em tom simpático e direto, em poucas linhas. Ela lê no celular: \
nada de JSON, nada de termos técnicos sem necessidade. O programa já envia junto a linha do tempo \
detalhada, então não repita todos os tempos, só os que você teve que adivinhar ou que mudaram."""


class AgentError(Exception):
    pass


def provider() -> str:
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if p in ("gemini", "anthropic"):
        return p
    if p:
        raise SystemExit(f"LLM_PROVIDER inválido: {p!r} (use 'gemini' ou 'anthropic')")
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise SystemExit("Nenhuma chave de IA configurada: preencha GEMINI_API_KEY ou ANTHROPIC_API_KEY no .env")


def model_name() -> str:
    return (_gemini_model or GEMINI_MODEL) if provider() == "gemini" else CLAUDE_MODEL


@functools.lru_cache(maxsize=1)
def _client():
    if provider() == "gemini":
        from google import genai
        from google.genai import types

        # v1beta: é onde ficam os apelidos (gemini-flash-latest) e a saída estruturada completa
        return genai.Client(http_options=types.HttpOptions(api_version="v1beta"))
    import anthropic

    return anthropic.Anthropic()


def _build_messages(session: Session, user_text: str) -> list[dict]:
    messages = [{"role": m["role"], "content": m["content"]} for m in session.history[-MAX_HISTORY:]]
    # As APIs exigem que a conversa comece com a usuária
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({
        "role": "user",
        "content": f"<estado>\n{session.state_text()}\n</estado>\n\n{user_text}",
    })
    return messages


def _respond_claude(messages: list[dict]) -> AgentResponse:
    response = _client().beta.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        # Se o modelo recusar o pedido, a API tenta de novo num modelo alternativo
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=messages,
        output_format=AgentResponse,
    )
    if response.stop_reason == "refusal":
        raise AgentError("O modelo recusou o pedido.")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise AgentError(f"Resposta incompleta (stop_reason={response.stop_reason}).")
    return response.parsed_output


_gemini_model: str | None = None


def _discover_gemini_model() -> str:
    """Escolhe o modelo Flash mais recente disponível para esta chave."""
    names = []
    for m in _client().models.list():
        name = (m.name or "").removeprefix("models/")
        actions = m.supported_actions or []
        if "flash" in name and "generateContent" in actions and not any(
            x in name for x in ("lite", "image", "tts", "audio", "live", "thinking", "exp")
        ):
            names.append(name)
    if not names:
        raise AgentError("nenhum modelo Gemini Flash disponível para esta chave")

    def version(n: str) -> tuple:
        nums = re.findall(r"\d+(?:\.\d+)?", n)
        return (float(nums[0]) if nums else 0.0, "preview" not in n, n)

    return max(names, key=version)


def _respond_gemini(messages: list[dict]) -> AgentResponse:
    global _gemini_model
    from google.genai import errors, types

    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=AgentResponse,
    )
    model = _gemini_model or GEMINI_MODEL
    try:
        try:
            response = _client().models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as e:
            if e.code != 404:
                raise
            # Modelo não existe (nomes mudam com o tempo): descobre um disponível e tenta de novo
            _gemini_model = _discover_gemini_model()
            log.warning("modelo %s não encontrado; usando %s", model, _gemini_model)
            response = _client().models.generate_content(model=_gemini_model, contents=contents, config=config)
    except errors.ClientError as e:
        if e.code == 429:
            raise AgentError("limite gratuito do Gemini atingido; espere um pouco e tente de novo") from e
        raise AgentError(f"erro do Gemini: {e.message or e}") from e
    except errors.ServerError as e:
        raise AgentError("o Gemini está instável agora; tente de novo em instantes") from e

    parsed = response.parsed
    if isinstance(parsed, AgentResponse):
        return parsed
    try:
        return AgentResponse.model_validate_json(response.text or "")
    except Exception as e:
        reason = response.candidates[0].finish_reason if response.candidates else "sem resposta"
        raise AgentError(f"resposta inválida do Gemini ({reason})") from e


def respond(session: Session, user_text: str) -> AgentResponse:
    messages = _build_messages(session, user_text)
    result = _respond_gemini(messages) if provider() == "gemini" else _respond_claude(messages)

    session.history.append({"role": "user", "content": user_text})
    session.history.append({"role": "assistant", "content": result.reply})
    return result

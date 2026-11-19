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
explica como quer a música para a coreografia, com palavras do dia a dia (ex.: "ir abaixando até sumir" é \
um fade, "uma entrando enquanto a outra sai" é crossfade, "troca de uma vez" é corte, "mais aguda" fala de tom). \
Você não ouve o áudio: você recebe a análise de cada faixa (duração, BPM, tom estimado, capítulos do YouTube, \
curva de energia, onde começa cada contagem de 8 e, quando existe, a letra com horários) e devolve um plano \
de edição que um programa executa.

Como trabalhar:
- Os tempos no plano são em segundos, sempre no tempo da música original de cada faixa.
- Ela costuma pedir duas coisas: juntar músicas e cortar trechos. Quando ela der minutos e segundos \
("de 1:20 até 2:45"), use exatamente esses tempos.
- Quando ela falar de partes da música sem dar tempo ("começa no refrão", "tira a introdução", "depois que \
cantam tal frase"), use primeiro a LETRA COM HORÁRIOS: frases marcadas com ↻ se repetem e quase sempre são o \
refrão; a introdução é o que vem antes da primeira frase cantada; "o fim do refrão" é a última frase daquele \
bloco de frases ↻. Os horários da legenda podem estar até ~1 s adiantados ou atrasados e a legenda automática \
pode errar palavras, então procure pelo sentido. Sem letra, estime pela curva de energia e pelos capítulos \
(refrões costumam ter energia alta e se repetir; introduções, energia baixa no começo). Em qualquer caso, diga \
na resposta onde você cortou citando a frase da letra (ex.: "começa em 'Ressuscita-me...'"), para ela conferir.
- Para começar um trecho numa frase cantada, use como início o começo da contagem de 8 imediatamente antes \
dessa frase (assim a dança entra no "1" e a primeira palavra não é cortada). Para terminar depois de uma \
frase, use o começo da contagem de 8 que vem logo depois que a frase termina.
- Padrões, a menos que ela peça outra coisa: transição crossfade de 4 s, snap "phrase", fade_in 0, \
ending "fade" com fade_out de 3 s, speed 1.0, semitones 0, volume_db 0, gap_before 0. Use snap "none" só quando ela \
der um tempo exato que não deve ser mexido.
- Nunca mude tom (semitones) nem velocidade (speed) por conta própria, nem para "combinar" as músicas: \
isso deixa a voz artificial. Só mude quando ela pedir explicitamente.
- Escolha a transição pelo efeito que ela quer:
  * "crossfade" (uma entra enquanto a outra sai): quando os andamentos diferem até ~8%, o programa \
iguala o ritmo durante a mistura e troca os graves sozinho, então misturas de 4 a 8 s ficam boas. Com \
andamentos muito diferentes, use no máximo 2 s ou prefira outra transição.
  * "echo" (a anterior para com um eco no ritmo): ótimo para mudar de clima ou de andamento, e para \
pausas de troca de formação (com gap_before, o eco preenche o silêncio).
  * "fade" (abaixa até sumir) para saídas suaves; "cut" (troca direta, no "1" da contagem) para \
viradas fortes de coreografia.
- Para o final do medley: ending "fade" (abaixa até sumir, com fade_out em segundos) ou "echo" \
(para com eco), quando ela pedir um final marcante.
- Pausa entre músicas (troca de formação): gap_before com "fade", "cut" ou "echo".
- Para "versão de ensaio mais lenta", mude só speed (ex.: 0.85). Para subir o tom, use semitones.
- Ao alterar algo, parta do PLANO ATUAL e devolva o plano inteiro com a mudança, mantendo todo o resto \
exatamente igual (mesmos tempos, transições, tom e velocidade). Mude só o que ela pediu.
- Se ela pedir para voltar como estava antes, use o PLANO ANTERIOR. Ela também pode usar o comando \
/desfazer, que faz isso sozinho; você pode lembrar dele quando ela não gostar de uma mudança.
- Mensagens marcadas como "[conferência automática]" vêm do programa, não dela: corrija o que for apontado \
e não fale da conferência na resposta.
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
        return genai.Client(http_options=types.HttpOptions(
            api_version="v1beta",
            # Tenta de novo, com espera crescente, quando o Google responde "sobrecarregado"
            retry_options=types.HttpRetryOptions(
                attempts=3, initial_delay=2, max_delay=10, http_status_codes=[500, 502, 503, 504]
            ),
        ))
    import anthropic

    return anthropic.Anthropic()


def _build_messages(session: Session, user_text: str, followups: list[dict] = ()) -> list[dict]:
    messages = [{"role": m["role"], "content": m["content"]} for m in session.history[-MAX_HISTORY:]]
    messages.append({
        "role": "user",
        "content": f"<estado>\n{session.state_text()}\n</estado>\n\n{user_text}",
    })
    messages += list(followups)
    # As APIs exigem que a conversa comece com a usuária e alterne usuária/assistente
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


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


def _gemini_candidates() -> list[str]:
    """Modelos Flash disponíveis para esta chave, do melhor para o pior (versões 'lite' por último)."""
    names = []
    for m in _client().models.list():
        name = (m.name or "").removeprefix("models/")
        actions = m.supported_actions or []
        if "flash" in name and "generateContent" in actions and not any(
            x in name for x in ("image", "tts", "audio", "live", "thinking", "exp")
        ):
            names.append(name)

    def rank(n: str) -> tuple:
        nums = re.findall(r"\d+(?:\.\d+)?", n)
        return ("lite" not in n, float(nums[0]) if nums else 0.0, "preview" not in n, n)

    return sorted(names, key=rank, reverse=True)


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
    tried: list[str] = []
    response = None
    for _ in range(4):  # o modelo escolhido + até 3 alternativos
        try:
            response = _client().models.generate_content(model=model, contents=contents, config=config)
            break
        except errors.ClientError as e:
            if e.code == 429:
                raise AgentError("limite gratuito do Gemini atingido; espere um pouco e tente de novo") from e
            if e.code != 404:
                raise AgentError(f"erro do Gemini: {e.message or e}") from e
            reason = "não encontrado"
        except errors.ServerError:
            # O SDK já tentou de novo algumas vezes; o modelo está sobrecarregado
            reason = "sobrecarregado"
        tried.append(model)
        try:
            alternatives = [m for m in _gemini_candidates() if m not in tried]
        except Exception:
            alternatives = []
        if not alternatives:
            break
        log.warning("modelo %s %s; tentando %s", model, reason, alternatives[0])
        if reason == "não encontrado":
            _gemini_model = alternatives[0]  # troca de vez; sobrecarga é temporária
        model = alternatives[0]
    if response is None:
        raise AgentError("o Gemini está sobrecarregado agora; tente de novo em alguns minutos")

    parsed = response.parsed
    if isinstance(parsed, AgentResponse):
        return parsed
    try:
        return AgentResponse.model_validate_json(response.text or "")
    except Exception as e:
        reason = response.candidates[0].finish_reason if response.candidates else "sem resposta"
        raise AgentError(f"resposta inválida do Gemini ({reason})") from e


def respond(session: Session, user_text: str, followups: list[dict] = ()) -> AgentResponse:
    """Pede uma resposta à IA. Não altera o histórico: quem decide o que guardar é o core.

    followups: mensagens extras depois do pedido (ex.: a resposta anterior da IA e os problemas que a
    conferência automática achou nela), para a IA corrigir o próprio plano.
    """
    messages = _build_messages(session, user_text, followups)
    return _respond_gemini(messages) if provider() == "gemini" else _respond_claude(messages)

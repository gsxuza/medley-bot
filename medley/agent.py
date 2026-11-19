"""Transforma o pedido em linguagem natural num Plan (saída estruturada da API do Claude)."""
import anthropic

from .plan import AgentResponse
from .session import MAX_HISTORY, Session

MODEL = "claude-opus-5-5"

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


def respond(client: anthropic.Anthropic, session: Session, user_text: str) -> AgentResponse:
    messages = [{"role": m["role"], "content": m["content"]} for m in session.history[-MAX_HISTORY:]]
    # A API exige que a conversa comece com a usuária
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({
        "role": "user",
        "content": f"<estado>\n{session.state_text()}\n</estado>\n\n{user_text}",
    })

    response = client.beta.messages.parse(
        model=MODEL,
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
    result = response.parsed_output

    session.history.append({"role": "user", "content": user_text})
    session.history.append({"role": "assistant", "content": result.reply})
    return result

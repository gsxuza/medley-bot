"""Estrutura do plano de edição.

O agente (LLM) só produz/edita este plano. Quem mexe no áudio é o render.py.
Os mesmos modelos servem de schema para a saída estruturada da API.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    track: str = Field(description="ID da faixa, ex: 'T1'")
    start: float = Field(description="Início do trecho, em segundos, no tempo da música original")
    end: float = Field(description="Fim do trecho, em segundos, no tempo da música original")
    transition: Literal["crossfade", "cut", "fade", "echo"] = Field(
        description=(
            "Como este trecho entra depois do anterior (ignorado no primeiro trecho). "
            "crossfade: os dois se sobrepõem, um sumindo e o outro entrando (o programa iguala o ritmo "
            "e troca os graves sozinho quando os andamentos são parecidos). "
            "cut: corte seco. "
            "fade: o anterior abaixa até sumir, depois este começa. "
            "echo: o anterior para com um eco no ritmo, que continua por baixo do começo deste."
        )
    )
    transition_seconds: float = Field(
        description="Duração do crossfade ou do fade-out do trecho anterior, em segundos"
    )
    gap_before: float = Field(description="Silêncio antes deste trecho, em segundos (0 = nenhum)")
    semitones: int = Field(description="Transposição de tom em semitons (0 = original, 1 = meio tom acima)")
    volume_db: float = Field(description="Ajuste de volume em dB depois da equalização automática (0 = nenhum)")
    parts: Literal["full", "instrumental", "vocals", "drums", "no_drums"] = Field(
        default="full",
        description=(
            "Que partes da música tocam neste trecho: full = música original; instrumental = sem a voz; "
            "vocals = só a voz; drums = só a bateria; no_drums = sem a bateria. "
            "Diferente de 'full' só funciona se a separação estiver disponível."
        ),
    )


class Plan(BaseModel):
    segments: list[Segment]
    fade_in: float = Field(description="Fade-in no começo do medley, em segundos")
    fade_out: float = Field(description="Fade-out no final do medley, em segundos (usado quando ending = 'fade')")
    ending: Literal["fade", "echo"] = Field(
        default="fade",
        description="Como o medley termina: 'fade' = abaixa até sumir; 'echo' = para com um eco no ritmo",
    )
    snap: Literal["phrase", "beat", "none"] = Field(
        description=(
            "Como ajustar o início e o fim de cada trecho: 'phrase' = no começo da contagem de 8 "
            "mais próxima (recomendado para dança), 'beat' = na batida mais próxima, 'none' = tempo exato"
        )
    )
    speed: float = Field(description="Velocidade final sem mudar o tom (1.0 = normal, 0.85 = 85% para ensaio)")

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data):
        """Planos antigos tinham snap_to_beat (true/false)."""
        if isinstance(data, dict) and "snap_to_beat" in data and "snap" not in data:
            old = data["snap_to_beat"]
            data = {k: v for k, v in data.items() if k != "snap_to_beat"}
            data["snap"] = "beat" if old else "none"
        return data


class AgentResponse(BaseModel):
    reply: str = Field(description="Mensagem para a usuária, em português, curta e clara")
    action: Literal["render", "none"] = Field(
        description="'render' quando há um plano novo ou alterado para gerar o áudio; 'none' para só conversar"
    )
    plan: Plan | None = Field(description="O plano completo (não só a mudança) quando action = 'render'")

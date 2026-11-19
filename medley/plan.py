"""Estrutura do plano de edição.

O agente (LLM) só produz/edita este plano. Quem mexe no áudio é o render.py.
Os mesmos modelos servem de schema para a saída estruturada da API.
"""
from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    track: str = Field(description="ID da faixa, ex: 'T1'")
    start: float = Field(description="Início do trecho, em segundos, no tempo da música original")
    end: float = Field(description="Fim do trecho, em segundos, no tempo da música original")
    transition: Literal["crossfade", "cut", "fade"] = Field(
        description=(
            "Como este trecho entra depois do anterior (ignorado no primeiro trecho). "
            "crossfade: os dois se sobrepõem, um sumindo e o outro entrando. "
            "cut: corte seco. "
            "fade: o anterior abaixa até sumir, depois este começa."
        )
    )
    transition_seconds: float = Field(
        description="Duração do crossfade ou do fade-out do trecho anterior, em segundos"
    )
    gap_before: float = Field(description="Silêncio antes deste trecho, em segundos (0 = nenhum)")
    semitones: int = Field(description="Transposição de tom em semitons (0 = original, 1 = meio tom acima)")
    volume_db: float = Field(description="Ajuste de volume em dB depois da equalização automática (0 = nenhum)")


class Plan(BaseModel):
    segments: list[Segment]
    fade_in: float = Field(description="Fade-in no começo do medley, em segundos")
    fade_out: float = Field(description="Fade-out no final do medley, em segundos")
    snap_to_beat: bool = Field(
        description="Ajustar início e fim de cada trecho para a batida mais próxima (recomendado)"
    )
    speed: float = Field(description="Velocidade final sem mudar o tom (1.0 = normal, 0.85 = 85% para ensaio)")


class AgentResponse(BaseModel):
    reply: str = Field(description="Mensagem para a usuária, em português, curta e clara")
    action: Literal["render", "none"] = Field(
        description="'render' quando há um plano novo ou alterado para gerar o áudio; 'none' para só conversar"
    )
    plan: Plan | None = Field(description="O plano completo (não só a mudança) quando action = 'render'")

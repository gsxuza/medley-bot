"""Confere o plano antes de gerar o áudio.

- Problemas pequenos e óbvios são corrigidos aqui mesmo (ex.: "até 9:99" numa música de 4 min).
- Problemas de verdade voltam para a IA corrigir (ex.: faixa que não existe, trecho de meio segundo).
"""
from . import stems
from .audio import TrackInfo, fmt_time
from .plan import Plan

MIN_SEGMENT = 2.0       # segundos
MAX_SEMITONES = 4       # mais que isso a voz fica muito artificial
SPEED_RANGE = (0.5, 1.5)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def check(plan: Plan, tracks: dict[str, TrackInfo]) -> tuple[Plan, list[str], list[str]]:
    """Retorna (plano corrigido, ajustes automáticos, problemas que a IA precisa resolver)."""
    plan = plan.model_copy(deep=True)
    fixes: list[str] = []
    problems: list[str] = []

    if not plan.segments:
        problems.append("o plano não tem nenhum trecho")
        return plan, fixes, problems

    for i, seg in enumerate(plan.segments, 1):
        tr = tracks.get(seg.track)
        if tr is None:
            problems.append(
                f"trecho {i}: a faixa {seg.track} não existe (faixas disponíveis: {', '.join(tracks) or 'nenhuma'})"
            )
            continue
        name = f"trecho {i} ({seg.track})"

        if seg.start < 0:
            seg.start = 0.0
            fixes.append(f"{name}: início negativo virou 0:00")
        if seg.end > tr.duration:
            if seg.end - tr.duration > 1:
                fixes.append(f"{name}: fim {fmt_time(seg.end)} passava do fim da música; usei {fmt_time(tr.duration)}")
            seg.end = tr.duration
        if seg.start >= tr.duration - MIN_SEGMENT:
            problems.append(
                f"{name}: começa em {fmt_time(seg.start)}, mas a música só tem {fmt_time(tr.duration)}"
            )
        elif seg.end - seg.start < MIN_SEGMENT:
            problems.append(
                f"{name}: vai de {fmt_time(seg.start)} a {fmt_time(seg.end)}, curto demais "
                f"(mínimo {MIN_SEGMENT:.0f} s). Confira se início e fim não estão trocados"
            )

        if seg.parts != "full" and not stems.available():
            problems.append(
                f"{name}: separar voz e instrumentos não está instalado neste computador; use parts 'full' "
                "e avise que essa função ainda não está disponível"
            )
        if abs(seg.semitones) > MAX_SEMITONES:
            problems.append(
                f"{name}: mudança de tom de {seg.semitones:+d} semitons é grande demais e deixa a voz artificial "
                f"(máximo ±{MAX_SEMITONES}); use um valor menor ou outra transição"
            )
        for attr, lo, hi, label in (
            ("transition_seconds", 0.0, 15.0, "transição"),
            ("gap_before", 0.0, 10.0, "silêncio antes"),
            ("volume_db", -12.0, 6.0, "ajuste de volume"),
        ):
            v = getattr(seg, attr)
            if not lo <= v <= hi:
                setattr(seg, attr, _clamp(v, lo, hi))
                fixes.append(f"{name}: {label} de {v:g} ajustado para {getattr(seg, attr):g}")

    # Crossfade não pode ser maior que metade de nenhum dos dois trechos que ele junta
    for i in range(1, len(plan.segments)):
        prev, seg = plan.segments[i - 1], plan.segments[i]
        if seg.transition != "crossfade" or seg.gap_before > 0:
            continue
        limit = 0.5 * min(prev.end - prev.start, seg.end - seg.start)
        if limit > 0 and seg.transition_seconds > limit:
            fixes.append(
                f"trecho {i + 1}: transição de {seg.transition_seconds:g} s era longa demais para trechos tão curtos; "
                f"usei {limit:.1f} s"
            )
            seg.transition_seconds = round(limit, 1)

    for attr, lo, hi, label in (("fade_in", 0.0, 30.0, "fade-in"), ("fade_out", 0.0, 30.0, "fade-out")):
        v = getattr(plan, attr)
        if not lo <= v <= hi:
            setattr(plan, attr, _clamp(v, lo, hi))
            fixes.append(f"{label} de {v:g} s ajustado para {getattr(plan, attr):g} s")

    if not SPEED_RANGE[0] <= plan.speed <= SPEED_RANGE[1]:
        problems.append(
            f"velocidade {plan.speed:g} fora do permitido ({SPEED_RANGE[0]:g} a {SPEED_RANGE[1]:g}); "
            "lembre que 1.0 é a velocidade normal e 0.85 é 85%"
        )

    return plan, fixes, problems

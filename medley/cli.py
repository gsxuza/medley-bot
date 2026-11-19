"""Modo terminal, para testar sem o Telegram.

  python -m medley.cli chat                      # conversa no terminal (usa a IA)
  python -m medley.cli render plano.json a.mp3 b.mp3 -o saida.mp3   # só o motor, sem IA
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()


def cmd_chat(args) -> None:
    import anthropic

    from .core import handle
    from .session import Session

    client = anthropic.Anthropic()
    session = Session.load(DATA_DIR / "chats" / "terminal.json")
    if args.novo:
        session.reset()
    print("Converse como ela faria. Para adicionar um arquivo local: /arquivo caminho.mp3   |   /novo   |   /sair")
    while True:
        try:
            text = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/sair", "/exit"):
            break
        if text == "/novo":
            session.reset()
            print("(começando do zero)")
            continue
        files = []
        if text.startswith("/arquivo "):
            p = Path(text.split(" ", 1)[1].strip().strip('"'))
            files, text = [(p, p.stem)], ""
        r = handle(client, session, text, DATA_DIR, files, progress=lambda m: print(f"  … {m}"))
        print(f"\nbot> {r.reply}")
        if r.mp3:
            print(f"\n{r.timeline}\n🎧 {r.mp3}")


def cmd_render(args) -> None:
    from .audio import analyze, import_file
    from .plan import Plan
    from .render import describe, render

    tracks = {}
    for i, f in enumerate(args.files, 1):
        p = Path(f)
        wav, meta = import_file(p, DATA_DIR / "cache")
        t = analyze(f"T{i}", str(p), wav, meta)
        tracks[t.id] = t
        print(t.summary())
    plan = Plan.model_validate(json.loads(Path(args.plan).read_text()))
    tl = render(plan, tracks, Path(args.output))
    print("\n" + describe(tl) + f"\n🎧 {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="medley")
    sub = ap.add_subparsers(required=True)
    c = sub.add_parser("chat")
    c.add_argument("--novo", action="store_true")
    c.set_defaults(fn=cmd_chat)
    r = sub.add_parser("render")
    r.add_argument("plan")
    r.add_argument("files", nargs="+", help="arquivos de áudio; viram T1, T2... na ordem")
    r.add_argument("-o", "--output", default="medley.mp3")
    r.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

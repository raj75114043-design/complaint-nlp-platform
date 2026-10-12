"""Command line.

    python -m cnlp.cli data                       # regenerate data/complaints.jsonl
    python -m cnlp.cli train [--promote-if-better] # train a challenger, compare with the champion, update the registry
    python -m cnlp.cli process "complaint text" --date 2026-10-01
"""
from __future__ import annotations

import argparse
import json
import logging

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from cnlp.data.generator import ISSUES, write
from cnlp.model import load_champion, save, train
from cnlp.pipeline import load_rows, process


def test_f1(model, rows) -> dict:
    test = [r for r in rows if r["split"] == "test" and not r["label_other"]]
    mlb = MultiLabelBinarizer(classes=ISSUES).fit([[]])
    p = model.predict([r["text"] for r in test])
    y, yh = mlb.transform([r["issues"] for r in test]), mlb.transform([x["issues"] for x in p])
    unseen = [i for i, r in enumerate(test) if r["unseen_phrasing"]]
    return {"issue_f1_micro": round(float(f1_score(y, yh, average="micro")), 4),
            "issue_f1_unseen": round(float(f1_score(y[unseen], yh[unseen], average="micro")), 4)}


def main() -> None:
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(prog="cnlp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    t = sub.add_parser("train")
    t.add_argument("--promote-if-better", action="store_true")
    p = sub.add_parser("process")
    p.add_argument("text")
    p.add_argument("--date", default="2026-10-01")
    a = ap.parse_args()
    if a.cmd == "data":
        write()
    elif a.cmd == "train":
        rows = load_rows()
        challenger = train(rows)
        cm = test_f1(challenger, rows)
        champ = load_champion()
        champ_m = test_f1(champ, rows) if champ else None
        better = champ_m is None or cm["issue_f1_micro"] > champ_m["issue_f1_micro"]
        save(challenger, cm, promote=a.promote_if_better and better)
        print(json.dumps({"challenger": challenger.version, "metrics": cm, "champion": champ.version if champ else None,
                          "champion_metrics": champ_m, "promoted": a.promote_if_better and better}, indent=2))
    else:
        print(json.dumps(process(a.text, a.date), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

"""Fusão de probabilidades Gaussian: seleção no OOF dev, sem alterar o candidato.

Pós-processamento exploratório da confiança; não muda a temperatura das
iterações Gaussian. O holdout é avaliado somente para a regra escolhida no dev.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import experimentos_prompt_removal as research

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/experimentos_fusao_gaussiano"
POWERS = (.25, .5, 1., 2., 4.)


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def fuse(probabilities, config):
    logs = np.log(np.maximum(np.stack(probabilities), 1e-30))
    if config["method"] == "geometric":
        return softmax(logs.mean(axis=0), axis=1)
    p = softmax(logs * config["power"], axis=2).mean(axis=0)
    return p / p.sum(axis=1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "holdout.json").exists():
        raise RuntimeError("Experimento concluído; não reabrir seleção após holdout")
    configs = [{"method": "arithmetic", "power": power} for power in POWERS] + [{"method": "geometric"}]
    write("protocol.json", {"created_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
                            "configs": configs, "selection": "development OOF only",
                            "limit": "Follow-up exploratory experiment; historical holdout exposure and multiple comparisons exist."})
    df, y, classes, dev, holdout, views, sem, _ = research.load()
    with np.load(ROOT / "runs/experimentos_prompt_removal/inner_oof.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"], dev)
        np.testing.assert_array_equal(f["classes"], classes)
        parts = [f[f"{name}_d128"] for name in views]
        np.testing.assert_allclose(fuse(parts, {"method": "arithmetic", "power": 1.}), f["mean_d128"], atol=1e-7)
    grid = [{"config": config, **research.metrics(y[dev], fuse(parts, config), classes)} for config in configs]
    winner = max(grid, key=lambda row: row["balanced_accuracy"])
    # Persistir decisão ANTES de ler os labels da avaliação externa.
    write("selected.json", {"grid": grid, "selected": winner, "selection_rows": len(dev), "outer_labels_used_for_selection": False})
    base_score = next(row["balanced_accuracy"] for row in grid if row["config"] == {"method": "arithmetic", "power": 1.})
    if winner["balanced_accuracy"] <= base_score:
        write("holdout.json", {"evaluated": False, "reason": "No development gain", "selected": winner})
        print(json.dumps({"grid": grid, "holdout_evaluated": False}), flush=True)
        return
    groups = research.infer_groups(df.text.iloc[holdout].tolist(), sem[holdout])
    predictions, _, _ = research.fit_views_predict(views, dev, holdout, y, df.prompt_id.iloc[dev].to_numpy(), groups, classes, dimensions=(128,))
    base = predictions["mean_d128"]
    result = fuse([predictions[f"{name}_d128"] for name in views], winner["config"])
    report = {"selected": winner["config"], "development": winner,
              "baseline": research.expanded_metrics(y[holdout], base, classes),
              "candidate": research.expanded_metrics(y[holdout], result, classes),
              "paired": research.paired_prompt(y[holdout], result, base, df.prompt_id.iloc[holdout], classes),
              "exploratory": True, "production_candidate_changed": False}
    write("holdout.json", report)
    np.savez_compressed(OUT / "holdout_predictions.npz", indices=holdout, classes=classes.astype(str), baseline=base, candidate=result)
    print(json.dumps({"selected": winner, "baseline_holdout": report["baseline"]["balanced_accuracy"],
                      "candidate_holdout": report["candidate"]["balanced_accuracy"], "paired": report["paired"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

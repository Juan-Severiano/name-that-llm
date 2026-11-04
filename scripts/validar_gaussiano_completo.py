"""Auditoria OOF5 do candidato congelado, sem busca de hiperparâmetros.

A configuração foi selecionada em exploração anterior sobre a mesma base.
Este OOF refaz todos os ajustes por fold, mas não é nested CV independente
da seleção de arquitetura/camadas. Nenhum resultado deve ser chamado de LB.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import experimentos_prompt_removal as research

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gaussiano_validacao_completa"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"created_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(), "folds": 5,
                "dimension": 128, "temperature": 3., "cluster_cut": .60, "minimum_group_for_gaussian": 4,
                "tuning": False, "classes_uniform_quota": False,
                "limit": "Configuration and legacy layers were selected during historical exploration on this dataset; this is not independent nested CV."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    df, y, classes, _, _, views, sem, _ = research.load()
    truth = df.prompt_id.to_numpy()
    outputs = {k: np.full((len(df), len(classes)), np.nan) for k in ("baseline", "gaussian", "robust")}
    fold_ids = np.full(len(df), -1, dtype=int)
    summaries = []
    for fold, (train, valid) in enumerate(GroupKFold(5).split(df, y, truth)):
        assert not set(truth[train]) & set(truth[valid])
        inferred = research.infer_groups(df.text.iloc[valid].tolist(), sem[valid])
        predictions, _, _ = research.fit_views_predict(views, train, valid, y, truth[train], inferred, classes, dimensions=(128,))
        baseline = research.baseline_predict(df, views, train, valid, y)
        gaussian = predictions["mean_d128"]
        robust, mask = research.route_small_groups(baseline, gaussian, inferred)
        fold_ids[valid] = fold
        for key, p in (("baseline", baseline), ("gaussian", gaussian), ("robust", robust)):
            outputs[key][valid] = p
        report = {"fold": fold, "rows": len(valid), "prompts": len(np.unique(truth[valid])),
                  "gaussian_routes_fraction": float(mask.mean()),
                  **{key: research.metrics(y[valid], value[valid], classes)["balanced_accuracy"] for key, value in outputs.items()}}
        summaries.append(report)
        print(json.dumps(report), flush=True)
    assert np.all(fold_ids >= 0) and all(np.isfinite(p).all() for p in outputs.values())
    report = {"folds": summaries, "metrics": {key: research.expanded_metrics(y, p, classes) for key, p in outputs.items()},
              "gaussian_vs_baseline": research.paired_prompt(y, outputs["gaussian"], outputs["baseline"], truth, classes),
              "robust_vs_baseline": research.paired_prompt(y, outputs["robust"], outputs["baseline"], truth, classes),
              "limit": protocol["limit"], "gpu": False, "submission": False}
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(OUT / "oof.npz", ids=df.id.to_numpy(dtype=str), classes=classes.astype(str), fold_ids=fold_ids, **outputs)
    print(json.dumps({"metrics": {k: v["balanced_accuracy"] for k, v in report["metrics"].items()},
                      "gaussian_delta_ci95": report["gaussian_vs_baseline"]["delta_ci95"],
                      "robust_delta_ci95": report["robust_vs_baseline"]["delta_ci95"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

"""Monta somente os artefatos robustos e StyleDistance para um dataset privado.

Não inclui train/test, credenciais, modelos 4B, LoRA nem a pasta de experimentos.
Os quatro encoders existentes são referenciados por hashes, sem nova cópia.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/kaggle_gaussiano"
SLUG = "juanseveriano/name-that-llm-gaussiano"
ENCODERS = ("bge-small-en-v1.5", "roberta-base", "deberta-v3-base", "Qwen3-0.6B")


def file_info(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "candidate": "gaussian5_robust", "files": {}, "encoders": {}}
    for name, source in (("robusto", ROOT / "runs/gaussiano_candidato/robusto"),
                         ("styledistance", ROOT / "data/modelos/styledistance")):
        for src in sorted(source.rglob("*")):
            if not src.is_file() or ".cache" in src.parts:
                continue
            rel = Path(name) / src.relative_to(source)
            target = DEST / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            manifest["files"][rel.as_posix()] = file_info(target)
    for name in ENCODERS:
        source = ROOT / "data/modelos" / name
        manifest["encoders"][name] = {"files": {
            p.relative_to(source).as_posix(): file_info(p)
            for p in sorted(source.rglob("*")) if p.is_file() and ".cache" not in p.parts
        }}
    meta = json.loads((DEST / "robusto/metadados.json").read_text())
    manifest["provenance"] = {
        "train_ids_sha256": meta["treino_ids_sha256"], "classes": meta["classes"],
        "train_rows": meta["treino_linhas"],
        "styledistance_repo": "StyleDistance/styledistance",
        "styledistance_revision": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
        "styledistance_license": "MIT", "selection": "Local grouped validation; no leaderboard tuning",
        "base_encoders_dataset": "juanseveriano/name-that-llm-encoders",
        "routing": "inferred group size <4: mean5; otherwise Gaussian five views",
    }
    (DEST / "manifesto.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (DEST / "dataset-metadata.json").write_text(json.dumps({
        "title": "Name That LLM - Gaussian robusto", "id": SLUG,
        "licenses": [{"name": "other"}],
    }, ensure_ascii=False, indent=2) + "\n")
    (DEST / "README.md").write_text(
        "# Gaussian robusto para Name That LLM\n\n"
        "Dataset privado para inferência offline. Artefatos JSON/NPZ treinados nos dados oficiais; "
        "sem pickle e sem dependência de train.csv durante a execução. "
        "Os quatro encoders anteriores vêm de juanseveriano/name-that-llm-encoders.\n\n"
        "StyleDistance: https://huggingface.co/StyleDistance/styledistance, revisão "
        "b7df5f0b0480773c097ba3121d83ca32b71015ca, licença MIT declarada no model card "
        "preservado em styledistance/README.md. Os termos dos dados da competição continuam aplicáveis.\n\n"
        "O manifesto registra hashes e classes. Embeddings podem ser extraídos em batches; "
        "agrupamento/classificação recebem o teste completo. Nenhuma quota de classes ou cap de grupo.\n"
    )
    expected = set(manifest["files"]) | {"manifesto.json", "dataset-metadata.json", "README.md"}
    actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Arquivo inesperado/ausente no staging: {actual ^ expected}")
    if any(p.endswith((".env", ".joblib", ".csv", ".py", ".ipynb")) for p in actual):
        raise ValueError("Staging contém arquivo proibido")
    print(json.dumps({"dataset": SLUG, "files": len(actual),
                      "bytes": sum((DEST / p).stat().st_size for p in actual),
                      "private_on_upload": True, "encoders_reused": list(ENCODERS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

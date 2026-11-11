# Repository Guidelines

## Project Structure & Module Organization

This Python project detects vehicles and license plates, reads plate text with PaddleOCR, and classifies Kia models with Ultralytics YOLO.

- Root modules contain the application: `main.py`, `reconocedor.py`, `detector.py`, `lector_ocr.py`, and `clasificador_vehiculo.py`.
- `config.py` defines plate-model paths, input/output images, confidence thresholds, and crop margins.
- `preparar_dataset_v2.py`, `entrenar_clasificador_v2.py`, and `evaluar_modelo.py` implement the classification workflow.
- `pruebas/` contains diagnostic scripts; `imagenes/` and `videos/` hold media inputs.
- `dataset_autos*` holds datasets, `modelos/` holds weights, and `runs/` and `resultados/` hold generated outputs. OpenVINO exports also appear in model directories.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

- `.\myvenv\Scripts\Activate.ps1` activates the existing local environment, if available.
- `python main.py` processes the image configured in `config.py`; ensure `modelos/best.pt` and the configured image exist first.
- `python preparar_dataset_v2.py --help` lists dataset preparation options. Review these before rebuilding data.
- `python entrenar_clasificador_v2.py --epochs 120 --imgsz 256` trains against `dataset_autos_v2/` and writes training artifacts under `runs/`.
- `python evaluar_modelo.py --modelo modelos/auto_cls_v2.pt` evaluates classification on validation images.

There is no separate build step. The dependency file is named `requeriments.txt` and is currently empty; it does not reproduce the environment.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` modules/functions/variables, `PascalCase` classes, and uppercase constants. Preserve Spanish domain terminology used by existing modules. Prefer `pathlib.Path` paths relative to `__file__` over machine-specific absolute paths. Guard executable workflows with `if __name__ == "__main__":`. No formatter or linter configuration is provided.

## Testing Guidelines

Existing `pruebas/test_*.py` files are manual diagnostics; no automated test framework or coverage threshold is configured. Run relevant scripts explicitly, for example `python pruebas/test_detector.py`, after checking their hardcoded paths and required weights. For model changes, report per-class evaluation results and check for predictions collapsing into one class. Name additional checks `test_<behavior>.py`.

## Commit & Pull Request Guidelines

Git history is unavailable in this directory, so established commit conventions cannot be verified. Use short, imperative subjects describing one focused change. PRs should describe the behavior changed, validation commands and results, and required model/data versions; link relevant issues and include sample outputs for detection changes. Review staged files carefully: `.gitignore` is empty. Exclude local environments, private plate imagery, and unnecessary generated artifacts.
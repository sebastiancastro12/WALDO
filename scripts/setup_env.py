"""Configura el entorno de desarrollo de WALDO tras crear/recrear el venv.

Cubre dos cosas que `uv sync` no resuelve por sí solo:

1. Deja `src/` en el `sys.path` del venv, de modo que se puedan importar tanto
   el paquete `waldo` como los módulos sueltos del pipeline de denoising
   (`functions`, `wavelet_denoising`, `u_net_model`, ...), que se importan entre
   sí por nombre simple (p.ej. `from functions import *`).

   Nota: el editable install que genera uv (uv_build 0.11.x) no deja `src/` en
   el path de forma fiable en este venv (su `.pth` no se procesa de manera
   consistente), así que `import waldo` tampoco funciona. Para evitar esa
   fragilidad escribimos un `sitecustomize.py`, que el intérprete importa
   siempre al arrancar.

2. Instala el binding precompilado de `pysparse` (Sparse2D) desde `vendor/`.

Uso (tras cada `uv sync` que recree el venv):
    uv run python scripts/setup_env.py
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
VENDOR_DIR = REPO_ROOT / "vendor" / "pysparse"

# El binding vendado de pysparse se compiló para esta versión de Python.
EXPECTED_PY = (3, 10)

_SITECUSTOMIZE_MARKER = "# WALDO: añade src/ al sys.path (gestionado por scripts/setup_env.py)"


def ensure_src_on_path(site_packages: Path) -> None:
    sitecustomize = site_packages / "sitecustomize.py"
    content = (
        f"{_SITECUSTOMIZE_MARKER}\n"
        "import sys\n"
        f"_src = {str(SRC_DIR)!r}\n"
        "if _src not in sys.path:\n"
        "    sys.path.append(_src)\n"
    )
    sitecustomize.write_text(content, encoding="utf-8")
    print(f"[ok] sitecustomize.py -> {SRC_DIR}")


def install_pysparse(site_packages: Path) -> None:
    matches = sorted(VENDOR_DIR.glob("pysparse.cpython-*-darwin.so"))
    if not matches:
        print(f"[warn] No hay binding vendado en {VENDOR_DIR}; se omite pysparse.")
        return
    src = matches[0]
    dst = site_packages / src.name
    shutil.copy2(src, dst)
    print(f"[ok] {src.name} -> {dst}")


def verify() -> None:
    # El .pth surte efecto en el próximo arranque del intérprete; para verificar
    # en este proceso añadimos src manualmente.
    sys.path.insert(0, str(SRC_DIR))
    checks = ["waldo", "functions", "wavelet_denoising", "pysparse", "pycs"]
    ok = True
    for mod in checks:
        try:
            __import__(mod)
            print(f"[ok] import {mod}")
        except Exception as exc:  # pragma: no cover - diagnóstico en runtime
            ok = False
            print(f"[FAIL] import {mod}: {type(exc).__name__}: {exc}")
    if not ok:
        sys.exit("[error] Alguna importación falló; revisa los mensajes anteriores.")


def main() -> None:
    if sys.version_info[:2] != EXPECTED_PY:
        sys.exit(
            f"[error] Este entorno espera Python "
            f"{EXPECTED_PY[0]}.{EXPECTED_PY[1]} (el binding pysparse es para esa "
            f"versión), pero se ejecutó con "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Usa: uv run python scripts/setup_env.py"
        )

    site_packages = Path(sysconfig.get_path("purelib"))
    ensure_src_on_path(site_packages)
    install_pysparse(site_packages)
    verify()
    print("[done] Entorno configurado.")


if __name__ == "__main__":
    main()

# waldo

Proyecto de Python gestionado con [uv](https://docs.astral.sh/uv/).

## Requisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python `>=3.10,<3.14` (uv lo gestiona automáticamente vía `.python-version`)

### Dependencias externas (NO disponibles en PyPI)

El método de denoising por wavelets (`Denoiser3D-IFU`) necesita estas piezas:

- **CosmoStat / pycs**: ya está declarado en `pyproject.toml` con fuente git, así
  que `uv sync` lo instala automáticamente (requiere **CMake** en el sistema para
  el build). No hay que instalarlo a mano.
- **Sparse2D** (backend C++ de transformadas dispersas). En macOS (Apple Silicon)
  se instala con Homebrew:
  ```bash
  brew install sparse2d
  ```
  > Nota: si tu CMake es >= 4 la compilación falla por compatibilidad con
  > pybind11. Edita la fórmula (`brew edit cosmostat/science/sparse2d`) y añade
  > `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` a la llamada a `cmake`.
- **pysparse** (binding de Python de Sparse2D): se instala con el script de
  configuración (ver más abajo). Requisitos en runtime (los provee
  `brew install sparse2d`): `libcfitsio`, `libfftw3`, `libomp`. El binding está
  enlazado a esas dylibs en `/opt/homebrew/opt/...`, por lo que solo funciona en
  macOS Apple Silicon con Sparse2D instalado vía Homebrew.

## Configuración

1. Sincroniza el entorno (uv descarga Python, instala las dependencias de PyPI y
   compila `pycs` desde git):

   ```bash
   uv sync
   ```

2. Ejecuta el script de configuración. Hace dos cosas que `uv sync` no cubre:
   deja `src/` en el `sys.path` (vía `sitecustomize.py`, para que los módulos del
   pipeline se importen entre sí, p.ej. `from functions import *`) e instala el
   binding precompilado de `pysparse` desde `vendor/`:

   ```bash
   uv run python scripts/setup_env.py
   ```

   > Vuelve a ejecutar el paso 2 solo si recreas el venv desde cero (un `uv sync`
   > normal conserva ambas cosas).

## Uso

Ejecuta el comando del proyecto:

```bash
uv run waldo
```

O ejecuta cualquier script de Python dentro del entorno:

```bash
uv run python -c "import waldo; waldo.main()"
```

## Dependencias

Añadir una dependencia:

```bash
uv add <paquete>
```

Añadir una dependencia de desarrollo:

```bash
uv add --dev <paquete>
```

## Estructura

```
WALDO/
├── pyproject.toml      # Configuración del proyecto y dependencias
├── uv.lock             # Versiones bloqueadas (no editar a mano)
├── .python-version     # Versión de Python del proyecto
├── README.md
├── scripts/
│   └── setup_env.py    # Post-sync: sys.path (src) + binding pysparse
├── vendor/
│   └── pysparse/       # Binding pysparse precompilado (Python 3.10)
└── src/                # En el sys.path (ver scripts/setup_env.py)
    ├── waldo/          # Paquete del proyecto (punto de entrada main)
    ├── functions.py            # Utilidades comunes del pipeline
    ├── toy_cube_dataset.py     # Generador de cubos sintéticos
    ├── u_net_model.py          # Arquitectura 3D U-Net
    ├── unet_train.py           # Entrenamiento del U-Net
    ├── wavelet_denoising.py    # Denoising 2D-1D por wavelets (pysparse)
    ├── pca_denoising.py        # Denoising por PCA
    ├── ica_denoising.py        # Denoising por ICA
    ├── sim_w2246_test_stats.py # Análisis sobre datos simulados/reales
    └── binned_plots.py         # Gráficas comparativas
```

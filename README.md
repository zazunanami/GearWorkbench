# Gearbox Design Workbench

Gearbox Design Workbench is a PySide6-based educational application for exploring two-stage parallel helical gearbox sizing, shaft loading, bending response, and fatigue-oriented design checks.

This project is an educational engineering tool. It is not certified for real machine design, manufacturing, safety-critical decisions, or professional engineering approval. Results should be independently verified against trusted references and standards before any real-world use.

## Features

- Interactive desktop GUI for input parameters, force summaries, shaft response plots, and design tables
- Helical gear mesh force calculations for tangential, radial, and axial loading
- Shaft shear, bending moment, deflection, and slope post-processing
- Fatigue and yield-oriented sizing checks based on DE-Gerber-style screening
- Included public-safe technical figures and a sanitized portfolio report PDF

## Requirements

- Python 3.12 or newer
- `uv`
- A desktop environment capable of running Qt applications

## Setup

```bash
uv venv
uv sync
```

## Run

```bash
uv run gearbox-workbench
```

Alternative module entry point:

```bash
uv run python -m makelpro.gearbox_gui
```

## Tests

```bash
uv run python -m compileall src tests
uv run pytest -q
```

## Project Structure

```text
GearWorkbench/
  README.md
  LICENSE
  .gitignore
  pyproject.toml
  src/
    makelpro/
      __init__.py
      analysis.py
      gearbox_gui.py
      technic_draw.py
      fbd.py
      assets/
        gear.ico
  docs/
    figures/
    report/
  tests/
```

## Reports and Documents

This repository includes sanitized academic report materials for portfolio purposes. Personal identifiers, student information, private metadata, and local machine paths have been removed or redacted.

## Known Limitations

- The `Fa * r` overturning moment assumption remains an educational simplification and should not be treated as a validated design standard.
- The ratio convention used in the project should be interpreted carefully; users should confirm whether it matches the reduction or speed-increase convention they expect.
- Deeper numerical validation against standards, references, and production-grade shaft design workflows remains future work.
- Optional CAD generation helpers are intentionally excluded from this public release to keep installation lighter and avoid unnecessary heavy dependencies.

## Public Release Notes

- Binaries, build artifacts, virtual environments, and packaging leftovers are intentionally excluded.
- Only one canonical GUI source version is included in this release folder.
- Public-safe report material is provided as a sanitized PDF. Editable office files were not included because they require additional manual verification for safe public distribution.

## License

Copyright (c) 2026 Zazu Nanami

The source code in this repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

# Gearbox Design Workbench

Gearbox Design Workbench is a PySide6-based educational application for exploring two-stage parallel helical gearbox sizing, shaft loading, bending response, and fatigue-oriented design checks.

This project is an educational engineering tool. It is not certified for real machine design, manufacturing, safety-critical decisions, or professional engineering approval. Results should be independently verified against trusted references and standards before any real-world use.

## Features

- Interactive desktop GUI for input parameters, force summaries, shaft response plots, and design tables
- Helical gear mesh force calculations for tangential, radial, and axial loading
- Axial thrust directions from the helix hands and input rotation, with optional `Fa * r` couples in the shaft analysis
- Shaft speeds, transmitted power, and pitch-line velocities in the results window
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

- The optional `Fa * r` couples follow the selected helix hands and input rotation, but thrust bearings and axial preload are not checked; treat this option as an educational extension, not a validated design standard.
- Stage ratios are defined as d2/d3 and d4/d5, so the gearbox is a speed increaser: Shaft B turns i1 times and Shaft C i1 x i2 times faster than the input (see the Speeds and Power tab in the results window).
- Bearings are modelled as simple supports at the shaft ends (spans of 200/400/200 mm). The stepped-shaft drawings in `docs/figures` place the bearing seats differently, so they illustrate the layout rather than the analysed geometry.
- The sanitized report PDF and the figures in `docs/` were produced with an earlier version of the analysis (before the Shaft B load-direction and DE-Gerber corrections). Results from the current GUI supersede their numbers.
- Deeper numerical validation against standards, references, and production-grade shaft design workflows remains future work.
- Optional CAD generation helpers are intentionally excluded from this public release to keep installation lighter and avoid unnecessary heavy dependencies.

## Public Release Notes

- Binaries, build artifacts, virtual environments, and packaging leftovers are intentionally excluded.
- Only one canonical GUI source version is included in this release folder.
- Public-safe report material is provided as a sanitized PDF. Editable office files were not included because they require additional manual verification for safe public distribution.

## License

Copyright (c) 2026 Zazu Nanami

The source code in this repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

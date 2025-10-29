# XenoScore: Open-Source ML Scoring System for Pig Xenotransplantation

**XenoScore** is a Python-based, modular scoring system that predicts xenograft outcomes in pig-to-human xenotransplantation. 
It is designed for continuous learning from user-contributed datasets and easy extension as the field evolves.

## Key Features
- **Modular components** for pre- and post-transplant variables (infection, renal/cardiac function, humoral markers, complement, donor genetics, pCMV, etc.).
- **Config-driven**: add/remove components and change weights via simple YAML.
- **Two scoring modes**:
  1. Weighted-score engine (transparent, editable weights).
  2. ML model engine (learns weights/probabilities from data; supports calibration).
- **Easy data I/O**: upload CSV via Streamlit app or CLI; export predictions/download example datasets.
- **Validation-ready**: designed to fit on real-world xenotransplant datasets when available.

> ⚠️ **Important**: This software is a research tool and **not** a clinical device. Do not use to guide patient care without appropriate validation and regulatory approvals.

## Quick Start
```bash
# (optional) create env and install dependencies
pip install -r requirements.txt

# run the Streamlit app
streamlit run app/streamlit_app.py

# or use the CLI
python -m xenoscore.cli score --input examples/example_dataset.csv   --config configs/default_components.yaml --weights configs/weights.example.yaml   --out predictions.csv
```

## Project Structure
```
xeno_score/
  app/streamlit_app.py
  configs/
    default_components.yaml
    weights.example.yaml
  examples/example_dataset.csv
  src/xenoscore/
    __init__.py
    cli.py
    config.py
    schemas.py
    registry.py
    data/
      io.py
      validation.py
    components/
      __init__.py
      core.py
      patient.py
      immunology.py
      donor.py
    scoring/
      weighted.py
      model.py
    ml/
      featurize.py
      train.py
  tests/test_scoring.py
  requirements.txt
  README.md
  LICENSE
```

## License
MIT License (see `LICENSE`).

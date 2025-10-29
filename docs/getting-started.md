# Getting Started

## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Hızlı Başlangıç
```bash
streamlit run app/streamlit_app.py
# veya
xenoscore score --input examples/example_dataset.csv --weights configs/weights.example.yaml --out predictions.csv
```

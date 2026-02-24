# DistilBERT SST-2 Benchmark

Knowledge distillation experiment: compressing BERT-base into DistilBERT for sentiment analysis.

## Stack
- Teacher: `textattack/bert-base-uncased-SST-2`
- Student: `distilbert-base-uncased` (small student baseline)
- Dataset: GLUE SST-2
- Loss: `alpha * KD + (1-alpha) * CE`

## Project structure
- `src/mini_distill/losses.py` — reusable KD + total loss helpers
- `src/mini_distill/metrics.py` — model size helpers
- `scripts/distill_sst2_tiny.py` — training script
- `scripts/benchmark_models.py` — parent vs base vs distilled accuracy + size report
- `tests/` — unit tests
- `pyproject.toml` — package + pytest config

## Setup
```bash
git clone https://github.com/SyedAkramaIrshad/distillbert-sst2-benchmark.git
cd distillbert-sst2-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run tests
```bash
pytest -q
```

## Run mini distillation training
```bash
python scripts/distill_sst2_tiny.py \
  --epochs 2 \
  --train-size 2000 \
  --val-size 500 \
  --output-dir ./artifacts/distilled-bert-tiny-sst2
```

## Benchmark parent vs distilled (accuracy + size)
```bash
python scripts/benchmark_models.py \
  --max-samples 500 \
  --distilled-model-path ./artifacts/distilled-bert-tiny-sst2 \
  --out ./artifacts/benchmark_report.json
```

Report includes:
- parent/teacher accuracy
- base student accuracy
- distilled student accuracy
- parameter count
- approximate fp32 parameter size (MB)
- local disk size for distilled folder
- deltas vs parent

## Latest benchmark metrics (local CPU run, v2)
Source: `artifacts/benchmark_report_v2.json` (400 validation samples)

- **Teacher (textattack/bert-base-uncased-SST-2)**
  - Accuracy: **0.9150**
  - Params: **109,483,778**
  - FP32 param size: **417.65 MB**

- **Student base (distilbert-base-uncased, pre-distillation)**
  - Accuracy: **0.4725**
  - Params: **66,955,010**
  - FP32 param size: **255.41 MB**

- **Student distilled v2 (`artifacts/distilled-bert-tiny-sst2-v2`)**
  - Accuracy: **0.8425**
  - Params: **66,955,010**
  - FP32 param size: **255.41 MB**
  - On-disk size: **256.10 MB**

- **Distilled vs Teacher delta**
  - Accuracy gap: **-7.25 points**
  - Size reduction (fp32 param est): **162.23 MB** (~**38.8% smaller**)

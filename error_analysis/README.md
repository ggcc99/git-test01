# LASER Base Model – Error Analysis Graph

This directory contains a self-contained Python script that generates a
four-panel error-analysis figure for the **LASER base model** on video
question-answering datasets (default: NExT-QA validation split).

---

## What the graph shows

| Panel | Title | What it shows |
|-------|-------|---------------|
| **(A)** | Error Count per Question Type | Number of wrong answers and per-type accuracy (dual axis) |
| **(B)** | Error Distribution by Question Type | Pie chart – share of total errors contributed by each type |
| **(C)** | Root-Cause Breakdown of Errors | Horizontal bar – *why* the model gets questions wrong |
| **(D)** | Correct vs. Incorrect per Question Type | Stacked bar – correct/incorrect split per type |

### NExT-QA Question-Type Codes

| Code | Meaning |
|------|---------|
| TC | Temporal – Causal |
| TN | Temporal – Descriptive |
| DC | Descriptive – Causal |
| DL | Descriptive – Location |
| DO | Descriptive – Object |
| CH | Causal – How |
| CW | Causal – Why |
| CU | Causal – Understand |

---

## Quick start

### 1 – Install dependencies

```bash
pip install matplotlib numpy
```

### 2 – Run with built-in sample data

```bash
cd error_analysis
python generate_error_graph.py
# → saves laser_base_error_graph.png in the current directory
```

### 3 – Run with real model predictions

After reproducing LASER and running inference, save your model outputs as a
JSON file with the following structure:

```json
[
  {
    "question_id": "1234-0",
    "question_type": "TC",
    "prediction": "because the child was scared",
    "ground_truth": "because the dog barked"
  },
  ...
]
```

Then run:

```bash
python generate_error_graph.py \
    --predictions /path/to/predictions.json \
    --output      my_error_graph.png \
    --dataset     "NExT-QA (val)"
```

---

## Error-cause taxonomy

The root-cause categories used in panel (C) reflect the main failure modes
observed in video-language base models:

| Category | Typical cause |
|----------|--------------|
| Insufficient temporal understanding | Model cannot determine *when* events happen or their order |
| Wrong object / entity identified | Confusion between visually similar objects |
| Incorrect action / event recognized | Action label is wrong (e.g. "running" vs "walking") |
| Weak causal / logical reasoning | Model misses the *why* or *because* relationship |
| Spatial relationship confusion | Front/back, left/right, on top of, etc. |
| Counting / quantity error | Off-by-one or higher counts |
| Attribute (color / size) error | Wrong color, size, or other perceptual attribute |

To obtain the cause labels automatically, you can add an `"error_cause"` field
to each wrong prediction in your JSON file.  If that field is absent, the
script falls back to the built-in proportions from the sample data.

---

## Adapting to your hardware

The script produces a static PNG and has **no GPU requirement**.
All heavy computation (model inference) should be done separately; this
script only needs the saved prediction JSON.

If you want to change figure size or DPI, edit the constants near the top of
`generate_error_graph.py`:

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))   # width × height in inches
plt.savefig(output_path, dpi=150, ...)              # increase dpi for print
```

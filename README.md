## Evaluating large language models on video-based Theory of Mind tasks

This repository contains the code used to evaluate large language models (LLMs) on video-based mental state recognition tasks using a four-alternative forced-choice (4-AFC) paradigm.

### Authors

Edward Bullock; Carrie Allison; Charles Nduka; Marcin A. Radecki; Simon Braschi; Simon Baron-Cohen.

### Abstract

We evaluate five LLMs (Gemini 3 Pro, Gemini 3 Flash, GPT-5, GPT-5 Mini, Claude Opus 4.5) on two mental state recognition datasets, EU-Emotion (27 mental states) and Mindreading (359 mental states), using a four-alternative forced-choice paradigm.

## Repository structure

- **`models/llm_wrapper.py`**: LLM wrapper for five model variants across three providers (Google Gemini, OpenAI, Anthropic). Includes deterministic caching and retry logic for Gemini calls.
- **`models/preprocessing.py`**: Frame extraction (4 frames via OpenCV) and base64 audio encoding.
- **`experiments/run_evaluation.py`**: Main evaluation runner. Requires `--model`, `--dataset`, `--trials_file`, `--data_dir` (optionally `--cache_dir`, `--results_dir`, `--seed`).
- **`experiments/trial_generator.py`**: Generates and saves 4-AFC trial JSON files with stratified foil sampling (default `--seed 42`).
- **`analysis/statistical_analysis.py`**: Statistical tests and effect sizes (Wilson confidence intervals; binomial tests vs chance; two-proportion z-tests; Cohen's \(h\); Fisher's exact tests; Bonferroni correction; power analysis).
- **`analysis/per_emotion_analysis.py`**: Per-mental-state accuracy breakdown and cross-model summary.
- **`prompts/`**: Prompt templates. `multimodal_prompt.txt` (Gemini: video frames + audio) and `video_only_prompt.txt` (GPT/Claude: video frames only).
- **`tests/test_statistical_analysis.py`**: Pytest suite verifying reported statistics against the values used in the manuscript.

## Installation

### Requirements

- **Python**: 3.8+

### Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root with API credentials:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Usage

### Step 1 — Generate trials

```bash
python experiments/trial_generator.py \
  --dataset eu_emotion \
  --data_dir data/eu_emotion/ \
  --output_path experiments/trials/eu_emotion_trials.json \
  --seed 42
```

### Step 2 — Run evaluation

```bash
python experiments/run_evaluation.py \
  --model gemini-3-pro \
  --dataset eu_emotion \
  --trials_file experiments/trials/eu_emotion_trials.json \
  --data_dir data/ \
  --results_dir results/
```

### Step 3 — Run statistical analysis

```bash
python analysis/statistical_analysis.py
```

### Step 4 — Per-mental-state breakdown

```bash
python analysis/per_emotion_analysis.py
```

### Step 5 — Run tests

```bash
pytest tests/
```

## Reproducibility

- **Randomisation**: Trial generation and evaluation default to **seed=42** (configurable via `--seed` in `experiments/trial_generator.py` and `experiments/run_evaluation.py`).
- **Caching**: API responses are cached in **`cache/`** as JSON files, keyed deterministically by **model name + video path (+ audio path if used) + candidate labels + prompt template hash**. Cached responses are reused on re-runs to allow the evaluation pipeline to be resumed.
- **Model/API versioning**: Exact replication depends on provider-side model and API versioning. The experiments in the manuscript used model versions available in **February 2026**.
- **Failed calls**: Failed or unparseable API calls are **not cached**, allowing automatic retry on subsequent runs. Gemini calls additionally use exponential backoff retry logic within a single run.

## Data availability

The EU-Emotion and Mindreading stimuli are not included in this repository due to licensing restrictions. Researchers may request access via `autismresearchcentre.com` (free for research use).

Expected directory layout:

```text
data/
  eu_emotion/
    happy/clip_001.mp4
    sad/clip_002.mp4
    ...
  mindreading/
    absorbed/clip_001.mov
    ...
```

References:

- EU-Emotion dataset: O'Reilly, H., Pigat, D., Fridenson, S., Berggren, S., Tal, S., Golan, O., Bölte, S., Baron-Cohen, S., & Lundqvist, D. (2016). The EU-Emotion Stimulus Set: A validation study. *Behavior Research Methods*, 48(2), 567–576. `https://doi.org/10.3758/s13428-015-0601-4`
- Mindreading library: Baron-Cohen, S., Golan, O., Wheelwright, S., & Hill, J. J. (2004). *Mind Reading: The Interactive Guide to Emotions* [DVD-ROM]. Jessica Kingsley Publishers.

## Citation

Please cite the accompanying manuscript (journal and year to be updated upon publication):

```bibtex
@article{bullock_tom_video_llm_tbd,
  title   = {Evaluating Large Language Models on Video-Based Theory-of-Mind Tasks: Comparison to Autistic and Non-Autistic People},
  author  = {Bullock, Edward and Allison, Carrie and Nduka, Charles and Radecki, Marcin A. and Braschi, Simon and Baron-Cohen, Simon},
  journal = {TBD},
  year    = {TBD}
}
```

## License

MIT
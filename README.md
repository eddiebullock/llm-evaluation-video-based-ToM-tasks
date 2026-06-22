## Evaluating large language models on video-based Theory of Mind tasks

This repository contains the code used to evaluate large language models (LLMs) on video-based mental state recognition tasks using a four-alternative forced-choice (4-AFC) paradigm.

### Authors

Edward Bullock; Carrie Allison; Charles Nduka; Marcin A. Radecki; Simon Braschi; Simon Baron-Cohen.

### Abstract

We evaluate four LLMs (Gemini 3 Flash, GPT-5, GPT-5 Mini, Claude Opus 4.5) on two mental state recognition datasets, EU-Emotion (27 mental states) and Mindreading (357 mental states), using a four-alternative forced-choice paradigm. **Gemini 3 Flash** additionally supports modality ablations (video-only, audio-only, multimodal) on both datasets. **Gemini 3 Pro** was evaluated in pilot runs but excluded from the main analysis (`EXCLUDED_MODELS` in `analysis/study_config.py`) because MR video-only accuracy was effectively identical to Flash while full ablations would require prohibitive API time under Tier-1 rate limits.

Human comparisons use EU validation benchmarks only: O'Reilly et al. (2016) facial expression (63%, *n* = 1,231; 6-AFC) for video-only, and Lassalle et al. (2019) vocal expression (45.19%, *n* = 427) for audio-only. No multimodal human benchmark exists and is not reported.

Full statistical results are documented in [RESULTS_SECTION.md](RESULTS_SECTION.md) and [METHODS_STATISTICAL_ANALYSIS.md](METHODS_STATISTICAL_ANALYSIS.md). Auto-generated tables: [analysis_outputs/results_section_summary.md](analysis_outputs/results_section_summary.md).

## Repository structure

- **`models/llm_wrapper.py`**: LLM wrapper for model variants across three providers (Google Gemini, OpenAI, Anthropic). Deterministic caching, Gemini retry logic, provider-specific generation settings.
- **`models/preprocessing.py`**: Frame extraction (4 frames via OpenCV) and base64 audio encoding.
- **`experiments/run_evaluation.py`**: Main evaluation runner (`--model`, `--dataset`, `--condition`, `--trials_file`, `--data_dir`; optional `--cache_dir`, `--results_dir`, `--seed`).
- **`experiments/trial_generator.py`**: Generates 4-AFC trial JSON with stratified foil sampling (default `--seed 42`).
- **`experiments/mindreading_audio_resolver.py`**: MR audio resolution and T-stimulus to paired V-video mapping (`resolve_mr_v_video_from_t_stimulus`).
- **`analysis/run_study_analysis.py`**: **Main analysis entrypoint** — loads results, applies MR fair subset, runs statistics and per-mental-state breakdowns.
- **`analysis/statistical_analysis.py`**: Wilson CIs, binomial vs chance, two-proportion *z*-tests vs human benchmarks, Fisher pairwise tests, Bonferroni correction, Cohen's *h*, power analysis.
- **`analysis/mr_fair_subset.py`**: Restricts MR video-only cross-model comparisons to trials with non-empty `video_path` (581-trial fair intersection).
- **`analysis/load_results.py`**: Loads summary JSON + per-trial CSVs; applies MR fair subset.
- **`analysis/basic_complex_analysis.py`**, **`analysis/basic_complex_mapping.py`**: Basic vs complex mental-state analyses.
- **`analysis/per_emotion_analysis.py`**: Per-mental-state accuracy (Neutral excluded), intensity summary.
- **`analysis/study_config.py`**: Four-model main study config and `EXCLUDED_MODELS`.
- **`prompts/`**: `multimodal_prompt.txt` (Gemini: frames + audio), `video_only_prompt.txt` (GPT/Claude: frames only).
- **`scripts/`**: Shell helpers for detached evaluation runs and video-only reruns.
- **`run_ablation_suite.py`**: Gemini 3 Flash modality ablation runner.
- **`tests/`**: Pytest suite including MR fair-subset logic and manuscript-aligned statistics.

## Key published numbers

### EU video-only (*n* = 118)

| Model | Accuracy |
|-------|----------|
| Gemini 3 Flash | 74.58% |
| GPT-5 | 73.73% |
| Claude Opus 4.5 | 72.03% |
| GPT-5 Mini | 67.80% |

### Mindreading video-only fair subset (*n* = 581)

657 MR trials use T-marker audio-only `.mov` files. In `video_only`, GPT/Claude/Mini skip trials with no video; legacy Flash runs counted text-only guesses on those trials, inflating *n* and headline accuracy. Cross-model MR video-only comparisons therefore use the **581-trial intersection** where all four models received non-empty video (see `analysis/mr_fair_subset.py`).

| Model | Accuracy |
|-------|----------|
| GPT-5 & Gemini 3 Flash | 65.40% |
| GPT-5 Mini | 62.82% |
| Claude Opus 4.5 | 57.49% |

### vs human (EU, two-sided *z*-tests)

- Flash video > O'Reilly facial 63%: *p* = .012
- GPT-5 video > 63%: *p* = .020
- Claude: *p* = .051 (marginal)
- Mini: n.s. (*p* = .301)
- Flash audio > Lassalle 45.19%: *p* = .011

**Caveat:** Human benchmarks used 6-AFC; this study uses 4-AFC — direct comparability is limited.

### Gemini 3 Flash modality ablations (MR video fair subset)

| Condition | Accuracy |
|-----------|----------|
| Video | 65.40% |
| Audio | 76.80% |
| Multimodal | 85.32% |

## Installation

### Requirements

- **Python**: 3.8+

### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

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
  --model gemini-3-flash \
  --dataset eu_emotion \
  --condition video_only \
  --trials_file experiments/trials/eu_emotion_trials.json \
  --data_dir data/ \
  --results_dir results/full_run \
  --cache_dir cache/full_run
```

Flash modality ablations (EU + Mindreading):

```bash
python run_ablation_suite.py
```

### Step 3 — Run study analysis (canonical)

After evaluations write to `results/full_run/`:

```bash
python analysis/run_study_analysis.py --results-dir results/full_run
```

Outputs in `analysis_outputs/`:

- `statistical_analysis.json`
- `per_emotion_breakdown.csv`
- `intensity_summary.csv`
- `basic_complex_summary.csv`
- `results_section_summary.md`

### Step 4 — Run tests

```bash
pytest tests/
```

## Reproducibility

- **Randomisation**: Trial generation and evaluation default to **seed=42**.
- **Caching**: API responses cached in `cache/` as JSON, keyed by model + video path (+ audio if used) + candidate labels + prompt hash. Failed calls are not cached.
- **Model/API versioning**: Experiments used model versions available in **February 2026**; exact replication depends on provider-side versioning.
- **MR fair subset**: Analysis applies `apply_mr_fair_video_subset()` automatically; optional pipeline hook `resolve_mr_v_video_from_t_stimulus()` in `run_evaluation.py` maps T-stimuli to paired V video (full rerun not required for published MR video-only tables).

## Data availability

EU-Emotion and Mindreading stimuli are not included due to licensing restrictions. Researchers may request access via [autismresearchcentre.com](https://www.autismresearchcentre.com) (free for research use).

Expected directory layout:

```text
data/
  eu_emotion/
    happy/clip_001.mp4
    ...
  mindreading/
    absorbed/clip_001.mov
    ...
```

Example summary JSONs (no per-trial CSVs) are in `results/example_summaries/`.

References:

- EU-Emotion: O'Reilly, H., et al. (2016). *Behavior Research Methods*, 48(2), 567–576. https://doi.org/10.3758/s13428-015-0601-4
- EU-Emotion Voice: Lassalle, A., et al. (2019). *Behavior Research Methods*, 51(2), 493–506.
- Mindreading: Baron-Cohen, S., et al. (2004). *Mind Reading: The Interactive Guide to Emotions* [DVD-ROM]. Jessica Kingsley Publishers.

## Citation

```bibtex
@article{bullock_tom_video_llm_tbd,
  title   = {Evaluating Large Language Models on Video-Based Theory-of-Mind Tasks},
  author  = {Bullock, Edward and Allison, Carrie and Nduka, Charles and Radecki, Marcin A. and Braschi, Simon and Baron-Cohen, Simon},
  journal = {TBD},
  year    = {TBD}
}
```

## License

MIT

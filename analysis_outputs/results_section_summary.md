# Results section — analysis summary

Generated from `analysis/run_study_analysis.py`.

## Mindreading video-only (fair cross-model subset)

Mindreading video_only cross-model comparisons use the intersection of trials where each model received a video stimulus (n=581). T-marker audio-only trials without video input are excluded.

| Model | Accuracy | n | Wilson 95% CI |
|-------|----------|---|---------------|
| claude-opus-4-5 | 57.49% | 581 | [53.43%, 61.44%] |
| gemini-3-flash | 65.40% | 581 | [61.45%, 69.16%] |
| gpt-5 | 65.40% | 581 | [61.45%, 69.16%] |
| gpt-5-mini | 62.82% | 581 | [58.82%, 66.66%] |

## EU-Emotions (all conditions)

| Model | Condition | Accuracy | n |
|-------|-----------|----------|---|
| claude-opus-4-5 | video_only | 72.03% | 118 |
| gemini-3-flash | audio_only | 58.47% | 118 |
| gemini-3-flash | multimodal | 81.36% | 118 |
| gemini-3-flash | video_only | 74.58% | 118 |
| gpt-5 | video_only | 73.73% | 118 |
| gpt-5-mini | video_only | 67.80% | 118 |

## Gemini Flash modality ablations

| Dataset | A | B | Acc A | Acc B | z | p (raw) |
|---------|---|---|-------|-------|---|---------|
| eu_emotion | audio_only | multimodal | 58.47% | 81.36% | -3.83 | 0.000127 |
| eu_emotion | audio_only | video_only | 58.47% | 74.58% | -2.62 | 0.00877 |
| eu_emotion | multimodal | video_only | 81.36% | 74.58% | 1.26 | 0.209 |
| mindreading | audio_only | multimodal | 76.80% | 85.32% | -5.44 | 5.453e-08 |
| mindreading | audio_only | video_only | 76.80% | 65.40% | 5.13 | 2.842e-07 |
| mindreading | multimodal | video_only | 85.32% | 65.40% | 9.72 | 2.447e-22 |

## vs human benchmark (EU)

- **claude-opus-4-5** video_only: z=1.95, p=0.05111, human=63.00% (n=1231)
- **gemini-3-flash** audio_only: z=2.56, p=0.01057, human=45.19% (n=427)
- **gemini-3-flash** video_only: z=2.50, p=0.01232, human=63.00% (n=1231)
- **gpt-5** video_only: z=2.32, p=0.02042, human=63.00% (n=1231)
- **gpt-5-mini** video_only: z=1.03, p=0.3014, human=63.00% (n=1231)

## Pairwise model comparisons (Fisher exact, Bonferroni)

### eu_emotion:video_only
- claude-opus-4-5 vs gemini-3-flash: OR=0.878, p_bonf=1, h=-0.06
- claude-opus-4-5 vs gpt-5: OR=0.918, p_bonf=1, h=-0.04
- claude-opus-4-5 vs gpt-5-mini: OR=1.223, p_bonf=1, h=0.09
- gemini-3-flash vs gpt-5: OR=1.045, p_bonf=1, h=0.02
- gemini-3-flash vs gpt-5-mini: OR=1.393, p_bonf=1, h=0.15
- gpt-5 vs gpt-5-mini: OR=1.333, p_bonf=1, h=0.13
### mindreading:video_only
- claude-opus-4-5 vs gemini-3-flash: OR=0.715, p_bonf=0.03992, h=-0.16
- claude-opus-4-5 vs gpt-5: OR=0.715, p_bonf=0.03992, h=-0.16
- claude-opus-4-5 vs gpt-5-mini: OR=0.800, p_bonf=0.4331, h=-0.11
- gemini-3-flash vs gpt-5: OR=1.000, p_bonf=1, h=0.00
- gemini-3-flash vs gpt-5-mini: OR=1.119, p_bonf=1, h=0.05
- gpt-5 vs gpt-5-mini: OR=1.119, p_bonf=1, h=0.05

## Manuscript reminders

- Remove autistic / Golan benchmark comparisons.
- EU human benchmarks: O'Reilly facial expression 63% (n=1231); Lassalle audio 45.19% (n=427); note 6-AFC vs 4-AFC and modality-match caveats.
- Exclude Neutral from per-mental-state summaries.
- Report MR audio/multimodal Flash results with spoken-label confound caveat.
- Basic vs complex: EU = six categories + low-intensity variants vs other non-neutral states; MR = exact label match only (no synonyms).

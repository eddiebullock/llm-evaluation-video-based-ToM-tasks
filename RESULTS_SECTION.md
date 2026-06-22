# Results

All models were evaluated using a four-alternative forced-choice (4-AFC) procedure; chance performance was 25%. One-sided binomial tests confirmed that every model performed significantly above chance on every reported battery and condition (all *p*s < .001). Unless stated otherwise, accuracies are reported with 95% Wilson score confidence intervals (CIs). Pairwise model comparisons used Fisher’s exact tests with Bonferroni correction (family-wise α = .005). Effect sizes are reported as Cohen’s *h* where applicable.

---

## Overall Model Performance

### EU-Emotions (video-only)

Four models were compared on the full EU-Emotions video-only battery (*N* = 118 trials per model; see Table 1). Accuracies ranged from 67.80% (GPT-5 Mini) to 74.58% (Gemini 3 Flash). No pairwise difference survived Bonferroni correction (all *p*s<sub>bonf</sub> = 1.00).

**Table 1**  
*EU-Emotions Video-Only Accuracy by Model*

| Model | *n* correct | *N* | Accuracy (%) | 95% CI |
|-------|-------------|-----|--------------|--------|
| Gemini 3 Flash | 88 | 118 | 74.58 | [66.03, 81.57] |
| GPT-5 | 87 | 118 | 73.73 | [65.13, 80.83] |
| Claude Opus 4.5 | 85 | 118 | 72.03 | [63.34, 79.34] |
| GPT-5 Mini | 80 | 118 | 67.80 | [58.92, 75.55] |

*Note.* All models received video input only. CIs = Wilson 95% confidence intervals.

### Mindreading (video-only)

Cross-model Mindreading video-only comparisons used the intersection of trials on which each model received a decodable video stimulus (*N* = 581 trials per model). Of 1,263 Mindreading trials, 657 referenced T-marker audio-only `.mov` files that could not be evaluated as video-only for GPT-5, GPT-5 Mini, and Claude Opus 4.5; Gemini 3 Flash had previously returned responses on some of these trials without video input, and those trials were excluded to ensure a comparable stimulus set. An additional 2 trials per model were excluded owing to video decode failure or unparseable model output.

On this fair subset, accuracies ranged from 57.49% (Claude Opus 4.5) to 65.40% (GPT-5 and Gemini 3 Flash; see Table 2). GPT-5 and Gemini 3 Flash achieved identical aggregate accuracy (380/581) but differed on 126 trials (McNemar discordant pairs: 51 favouring Gemini 3 Flash, 51 favouring GPT-5).

**Table 2**  
*Mindreading Video-Only Accuracy by Model (Fair Video-Evaluated Subset)*

| Model | *n* correct | *N* | Accuracy (%) | 95% CI |
|-------|-------------|-----|--------------|--------|
| GPT-5 | 380 | 581 | 65.40 | [61.45, 69.16] |
| Gemini 3 Flash | 380 | 581 | 65.40 | [61.45, 69.16] |
| GPT-5 Mini | 365 | 581 | 62.82 | [58.82, 66.66] |
| Claude Opus 4.5 | 334 | 581 | 57.49 | [53.43, 61.44] |

*Note.* *N* = 581 is the trial intersection across all four models with non-empty `video_path` and valid model output.

---

## Modality Ablations (Gemini 3 Flash)

Gemini 3 Flash was further evaluated under video-only, audio-only, and multimodal (audio + video) input conditions. Human benchmark comparisons were conducted for EU-Emotions only, using validation data from O’Reilly et al. (2016; facial expression, 63.00%, *N* = 1,231) and Lassalle et al. (2019; UK vocal expression, 45.19%, *N* = 427). O’Reilly et al. report separate benchmarks for facial expression (63%), body gesture (77%), and social scenes (72%); facial expression was selected as the closest match to full-face video clips in the present stimuli, although no single human modality perfectly matches the models’ full video presentation. Direct statistical comparability is further limited because validation studies used 6-AFC designs whereas the present study used 4-AFC.

### EU-Emotions

On EU-Emotions (*N* = 118 per condition), multimodal input yielded the highest accuracy (81.36%, 95% CI [73.38, 87.35]), followed by video-only (74.58%, 95% CI [66.03, 81.57]) and audio-only (58.47%, 95% CI [49.45, 66.96]; see Table 3). Multimodal accuracy exceeded audio-only accuracy (*z* = −3.83, *p*<sub>bonf</sub> < .001, *h* = −0.51). The difference between multimodal and video-only accuracy was not significant after Bonferroni correction (*z* = 1.26, *p*<sub>bonf</sub> = 1.00, *h* = 0.16). Audio-only accuracy was significantly lower than video-only accuracy at the uncorrected level (*z* = −2.62, *p* = .009) but not after Bonferroni correction (*p*<sub>bonf</sub> = .053).

Relative to human benchmarks, Gemini 3 Flash video-only accuracy (74.58%) was significantly higher than O’Reilly et al.’s (2016) facial-expression validation accuracy (63.00%, *N* = 1,231), *z* = 2.50, *p* = .012, *h* = 0.25. Audio-only accuracy (58.47%) was significantly higher than Lassalle et al.’s (2019) UK vocal-expression benchmark (45.19%, *N* = 427), *z* = 2.56, *p* = .011, *h* = 0.27. No direct human benchmark was available for multimodal EU-Emotions performance.

**Table 3**  
*Gemini 3 Flash EU-Emotions Accuracy by Input Modality*

| Condition | *n* correct | *N* | Accuracy (%) | 95% CI |
|-----------|-------------|-----|--------------|--------|
| Multimodal | 96 | 118 | 81.36 | [73.38, 87.35] |
| Video-only | 88 | 118 | 74.58 | [66.03, 81.57] |
| Audio-only | 69 | 118 | 58.47 | [49.45, 66.96] |

### Mindreading

On Mindreading, performance was strongly modality dependent (see Table 4). Multimodal accuracy was highest (85.32%, *N* = 1,240 valid), followed by audio-only (76.80%, *N* = 1,263) and video-only on the fair subset (65.40%, *N* = 581). All pairwise modality comparisons were significant after Bonferroni correction (all *p*s<sub>bonf</sub> < .001). The largest effect was between multimodal and video-only performance (*z* = 9.72, *p*<sub>bonf</sub> < .001, *h* = 0.47).

Audio-inclusive conditions are interpreted cautiously. Mindreading clips may contain spoken content that is directly diagnostic of the target mental-state label; elevated audio-only and multimodal accuracies may therefore partly reflect spoken-label confounds rather than superior mentalising from prosody or integrated audiovisual cues alone.

**Table 4**  
*Gemini 3 Flash Mindreading Accuracy by Input Modality*

| Condition | *n* correct | *N* | Accuracy (%) | 95% CI |
|-----------|-------------|-----|--------------|--------|
| Multimodal | 1,058 | 1,240 | 85.32 | [83.24, 87.18] |
| Audio-only | 970 | 1,263 | 76.80 | [74.39, 79.05] |
| Video-only (fair subset) | 380 | 581 | 65.40 | [61.45, 69.16] |

*Note.* Video-only *N* reflects the cross-model fair subset (see Mindreading section above). Audio-only and multimodal *N*s reflect all valid trials in those conditions.

---

## Comparison to Human Benchmarks

Human benchmark comparisons were restricted to EU-Emotions and are reported within the modality ablation section above. In summary, on video-only EU-Emotions, Gemini 3 Flash (*z* = 2.50, *p* = .012, *h* = 0.25) and GPT-5 (*z* = 2.32, *p* = .020, *h* = 0.23) performed significantly above O’Reilly et al.’s (2016) facial-expression benchmark (63.00%). Claude Opus 4.5 showed a marginal advantage (72.03% vs. 63.00%, *z* = 1.95, *p* = .051, *h* = 0.19). GPT-5 Mini did not differ significantly from the facial-expression benchmark (67.80% vs. 63.00%, *z* = 1.03, *p* = .301, *h* = 0.10). On audio-only EU-Emotions, Gemini 3 Flash exceeded Lassalle et al.’s (2019) vocal benchmark (*p* = .011). No Mindreading human benchmarks were analysed.

**Table 5**  
*EU-Emotions Video-Only Model Performance Relative to O’Reilly et al. (2016) Facial-Expression Benchmark (63.00%, N = 1,231)*

| Model | Accuracy (%) | *z* | *p* | Cohen’s *h* |
|-------|--------------|-----|-----|-------------|
| Gemini 3 Flash | 74.58 | 2.50 | .012 | 0.25 |
| GPT-5 | 73.73 | 2.32 | .020 | 0.23 |
| Claude Opus 4.5 | 72.03 | 1.95 | .051 | 0.19 |
| GPT-5 Mini | 67.80 | 1.03 | .301 | 0.10 |

*Note.* Two-sided two-proportion *z*-tests. Human benchmark = raw facial-expression accuracy from O’Reilly et al. (2016); original human task used 6-AFC.

---

## Pairwise Model Comparisons

### EU-Emotions (video-only)

Among the six pairwise comparisons (four models), no difference survived Bonferroni correction (all *p*s<sub>bonf</sub> = 1.00). The largest uncorrected difference was between Gemini 3 Flash and GPT-5 Mini (odds ratio [OR] = 1.39, *p* = .314, *h* = 0.15).

### Mindreading (video-only, fair subset)

On the 581-trial fair subset, Claude Opus 4.5 performed significantly below both GPT-5 and Gemini 3 Flash after Bonferroni correction (OR = 0.72, *p*<sub>bonf</sub> = .040, *h* = −0.16 for both comparisons). Claude Opus 4.5 did not differ significantly from GPT-5 Mini (*p*<sub>bonf</sub> = .433). Gemini 3 Flash and GPT-5 did not differ (OR = 1.00, *p*<sub>bonf</sub> = 1.00). GPT-5 and GPT-5 Mini did not differ (*p*<sub>bonf</sub> = 1.00).

**Table 6**  
*Pairwise Model Comparisons on Mindreading Video-Only (Fair Subset; Bonferroni-Corrected)*

| Model A | Model B | OR | *p* (raw) | *p* (Bonferroni) | Cohen’s *h* |
|---------|---------|-----|-----------|------------------|-------------|
| Claude Opus 4.5 | Gemini 3 Flash | 0.72 | .007 | .040 | −0.16 |
| Claude Opus 4.5 | GPT-5 | 0.72 | .007 | .040 | −0.16 |
| Claude Opus 4.5 | GPT-5 Mini | 0.80 | .072 | .433 | −0.11 |
| Gemini 3 Flash | GPT-5 | 1.00 | 1.000 | 1.000 | 0.00 |
| Gemini 3 Flash | GPT-5 Mini | 1.12 | .392 | 1.000 | 0.05 |
| GPT-5 | GPT-5 Mini | 1.12 | .392 | 1.000 | 0.05 |

---

## Per-Mental-State Analyses

*Neutral* was excluded from all per-mental-state summaries because it denotes affective neutrality rather than a mental state (O’Reilly et al., 2016).

### EU-Emotions (video-only, four models)

Recognition accuracy varied markedly across mental states. Averaged across the four video-only models, the most difficult states included *Unfriendly* (*M* = 12.5%), *Jealous* (*M* = 25.0%), *Interested* (*M* = 25.0%), and *Angry* (*M* = 25.0%). The easiest states included *Sad*, *Sad Low Intensity*, *Disgusted*, and *Worried* (all *M* = 100%). Full per-state accuracies are provided in Supplementary Materials (`per_emotion_breakdown.csv`).

### Mindreading (video-only, four models)

On the 581-trial fair video subset, per-mental-state accuracy was highly variable across the 360 represented states. Many states were recognised at ceiling by all four models (e.g., *glad*, *baffled*, *jaded*); others were at floor (e.g., *committed*, *lured*, *selfish*). This heterogeneity is consistent with the broader Mindreading battery sampling rare and subtle mental-state concepts.

For Gemini 3 Flash, modality-related patterns mirrored the aggregate findings: audio-inclusive conditions yielded higher per-state accuracies than video-only for most states, consistent with the spoken-label confound noted above.

### High versus low intensity (EU-Emotions)

EU-Emotions includes explicit low-intensity variants (e.g., *Happy Low Intensity* vs. *Happy*). Paired comparisons across intensity-matched base labels yielded mixed results (see Table 7). GPT-5 Mini showed higher accuracy on high-intensity than low-intensity variants (*M*<sub>diff</sub> = +10.61 percentage points across 11 pairs). In contrast, Gemini 3 Flash (*M*<sub>diff</sub> = −7.92 pp, 12 pairs), Claude Opus 4.5 (−4.55 pp, 11 pairs), and GPT-5 (−3.03 pp, 11 pairs) showed small advantages for low-intensity variants.

**Table 7**  
*High- Versus Low-Intensity Accuracy on EU-Emotions (Paired Base Labels)*

| Model | Pairs | *M* high-intensity (%) | *M* low-intensity (%) | *M* difference (pp) |
|-------|-------|------------------------|----------------------|---------------------|
| GPT-5 Mini | 11 | 89.39 | 78.79 | +10.61 |
| GPT-5 | 11 | 87.88 | 90.91 | −3.03 |
| Claude Opus 4.5 | 11 | 75.76 | 80.30 | −4.55 |
| Gemini 3 Flash | 12 | 72.63 | 80.56 | −7.92 |

*Note.* Positive values indicate higher accuracy on high-intensity base labels. Neutral excluded.

### Basic versus complex mental states

Basic emotions were operationalised as the six Ekman-style categories: happiness, sadness, fear, anger, disgust, and surprise. On EU-Emotions, basic-emotion trials comprised the 12 labels corresponding to the six base categories and their low-intensity variants (49 trials per model); all remaining non-neutral states were classified as complex (61 trials per model). This classification was confirmed a priori.

On EU-Emotions video-only, basic-emotion accuracy was descriptively higher than complex-emotion accuracy for GPT-5 (85.7% vs. 60.7%), GPT-5 Mini (75.5% vs. 57.4%), and Claude Opus 4.5 (79.6% vs. 63.9%). Gemini 3 Flash showed comparable basic and complex accuracy (73.5% vs. 72.1%; see Table 8). Formal inferential tests for this contrast were not conducted; see Discussion for interpretive limitations.

On Mindreading, basic emotions were defined by exact label match to the six canonical English terms only (*happy*, *sad*, *afraid*, *angry*, *disgusted*, *surprised*); near-synonyms (e.g., *fearful*, *terrified*, *scared*) were not included. Only eight fair-subset trials met this criterion (*happy*, *n* = 2; *sad*, *n* = 2; *angry*, *disgusted*, *surprised*, *n* = 1 each; *afraid*, *n* = 0). Basic-emotion accuracy was therefore near ceiling for GPT-5 and GPT-5 Mini (100%) and descriptively higher than complex-emotion accuracy (56.9%–64.9%), but the basic-emotion cell is underpowered and should be interpreted cautiously.

**Table 8**  
*Basic Versus Complex Mental-State Accuracy (Video-Only)*

| Dataset | Model | Basic % (*n*/N) | Complex % (*n*/N) |
|---------|-------|-----------------|-------------------|
| EU-Emotions | Gemini 3 Flash | 73.5 (36/49) | 72.1 (44/61) |
| EU-Emotions | GPT-5 | 85.7 (42/49) | 60.7 (37/61) |
| EU-Emotions | GPT-5 Mini | 75.5 (37/49) | 57.4 (35/61) |
| EU-Emotions | Claude Opus 4.5 | 79.6 (39/49) | 63.9 (39/61) |
| Mindreading† | Gemini 3 Flash | 87.5 (7/8) | 64.9 (373/575) |
| Mindreading† | GPT-5 | 100.0 (8/8) | 64.9 (372/573) |
| Mindreading† | GPT-5 Mini | 100.0 (8/8) | 62.3 (358/575) |
| Mindreading† | Claude Opus 4.5 | 87.5 (7/8) | 56.9 (327/575) |

*Note.* EU basic = six categories plus low-intensity variants (confirmed classification). Mindreading basic = exact matches to the six canonical labels only (no synonyms); *N* = 8 trials total in the fair video subset.

---

## Cross-Dataset Generalisation (Video-Only)

All four models showed numerically higher accuracy on EU-Emotions than on Mindreading under video-only input (see Table 9). This difference was statistically significant for Claude Opus 4.5 (72.03% vs. 57.49%; difference = −14.55 pp, *z* = 2.94, *p*<sub>bonf</sub> = .013, *h* = 0.31). For Gemini 3 Flash (−9.17 pp, *p* = .053, *p*<sub>bonf</sub> = .214), GPT-5 (−8.32 pp, *p* = .080, *p*<sub>bonf</sub> = .320), and GPT-5 Mini (−4.97 pp, *p* = .306, *p*<sub>bonf</sub> = 1.000), EU–Mindreading differences did not reach significance after Bonferroni correction.

**Table 9**  
*Cross-Dataset Video-Only Performance (EU-Emotions vs. Mindreading)*

| Model | EU accuracy (%) | MR accuracy (%) | Difference (pp) | *p* (raw) | *p* (Bonferroni) |
|-------|-----------------|-----------------|-----------------|-----------|------------------|
| Claude Opus 4.5 | 72.03 | 57.49 | −14.55 | .003 | .013 |
| Gemini 3 Flash | 74.58 | 65.40 | −9.17 | .053 | .214 |
| GPT-5 | 73.73 | 65.40 | −8.32 | .080 | .320 |
| GPT-5 Mini | 67.80 | 62.82 | −4.97 | .306 | 1.000 |

*Note.* Mindreading accuracies are from the 581-trial fair subset. EU *N* = 118; Mindreading *N* = 581.

---

## References (cited in Results)

O’Reilly, H., Pigat, D., Fridenson, S., Berggren, S., Tal, S., Golan, O., … Lundqvist, D. (2016). The EU-Emotion Stimulus Set: A validation study. *Behavior Research Methods*, *48*(2), 567–576.

Lassalle, A., O’Reilly, H., Rohlfing, K. J., Fridenson-Hayo, S., Berggren, S., Tal, S., … Golan, O. (2019). The EU-Emotion Voice Database. *Behavior Research Methods*, *51*(2), 493–506.

---

*Analysis source: `analysis_outputs/statistical_analysis.json` (generated 15 June 2026). Reproduce with: `python analysis/run_study_analysis.py --results-dir results/full_run`.*

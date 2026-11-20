# Deep Past: what the submission history actually shows

Dhruv Bansal · retrospective prepared 21 November 2026

The Deep Past Challenge asked participants to translate Akkadian transliteration into English. My Kaggle profile records a final position of 481 out of 2,674 teams (roughly the top 18%). The useful engineering lesson from my recovered submission history is that a more elaborate inference pipeline did not necessarily score better.

## Recorded results

These are Kaggle submission scores retrieved from my account on 21 November 2026, not newly reproduced experiments. Blank scores in the export are omitted; they are not interpreted as zero. The export does not identify which submissions were selected for the final leaderboard.

| Submitted (Kaggle timestamp) | Recorded description | Public score | Private score |
| --- | --- | ---: | ---: |
| 18 Mar 2026 | DeepPastChallenge, version 3 | 35.7071 | 35.7355 |
| 20 Mar 2026 | DeepPastChallenge3, version 6 | 1.1762 | 1.2274 |
| 21 Mar 2026 | Fork of DeepPastChallenge, version 2 | 0.2638 | 0.2226 |
| 23 Mar 2026 | Description absent (two submissions) | 35.3371 | 34.9161 |
| 23 Mar 2026 | TTA DeepPastChallenge, version 3 | 33.9979 | 34.0462 |

The TTA submission's private score was 1.6893 points below the earlier 35.7355 result. This comparison is observational: several settings changed together, so it does not isolate the effect of TTA.

## Pipeline visible in the recovered source

The notebooks load existing ByT5 checkpoints from community model sources, normalize transliteration, generate candidate English translations, normalize outputs, and choose a candidate using a consensus score. The selection combines pairwise chrF++, BLEU and token-set overlap, plus a length preference. Candidate generation includes beam search and sampling.

The later TTA notebook uses two prompt prefixes and two input variants for each of two checkpoints, producing eight candidate pools. Its transformed input removes several punctuation and gap characters. It also prefers candidates whose gap counts approximately match a heuristic estimate from the source. These are mechanisms visible in the source; the recovered notebooks contain no saved execution outputs establishing how often each path ran.

The code uses community checkpoints, including [Assia Ben's final ByT5 dataset](https://www.kaggle.com/datasets/assiaben/final-byt5) and [Mattia Angeli's ByT5 Akkadian MBR model](https://www.kaggle.com/models/mattiaangeli/byt5-akkadian-mbr-v2). Their training and pretrained weights are not my original work. The notebook comments also refer to host guidance and community postprocessing. I do not claim authorship of every inherited code block or a from-scratch translation model. A full upstream diff is not available in this retrospective.

## Failure modes worth discussing

1. **Complexity without an isolated comparison.** Prompt changes, transformed inputs, candidate generation and reranking changed together. The resulting score cannot tell us which component helped or hurt.
2. **Destructive preprocessing.** Removing gap or punctuation markers can remove information a byte-level model learned to use. That is a plausible risk, not a proven explanation of the score drop.
3. **Silent fallback output.** The recovered code can produce a generic damaged-tablet translation after inference errors. A valid CSV can therefore hide model failures. The very low submissions deserve investigation, but the available evidence does not establish their cause.
4. **Conditional inference.** The later source disables sampling when the test set exceeds 500 rows. Behavior should be reported for the actual run, with the branch logged and its effect measured.
5. **Consensus is not correctness.** Similar candidate translations can agree on the same error. Pairwise overlap is a heuristic, not a substitute for held-out references.

## What I would change in a fresh experiment

Freeze a development split before optimizing. Keep a single checkpoint and fixed decoding as the baseline. Change only one factor at a time: normalization, second checkpoint, candidate diversity, TTA and reranking. Save per-example candidates, selected outputs, failure counts, environment versions and exact checkpoint identities. Fail visibly on inference errors or explicitly count every fallback. Select with development metrics, then evaluate once on held-out data.

Those practices are now explicit design requirements for the customer-support intent-routing benchmark in this portfolio: grouped data splits, validation-only selection, recorded artifacts and a separate challenge set.

## Evidence and limits

- Final rank: [Kaggle profile](https://www.kaggle.com/dhruvb2028), observed 21 November 2026.
- Competition: [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation).
- Submission history: authenticated Kaggle CLI export, retained privately. Scores above are transcribed from that export.
- Sources requested through Kaggle CLI: `dhruvb2028/deeppastchallenge/3` and `dhruvb2028/tta-deeppastchallenge/3`; the downloaded source files are retained privately, with SHA-256 hashes in the evidence manifest. Returned kernel metadata may describe the current notebook; it is not treated as proof of historical settings. The recovered source has not been independently linked to the exact settings that generated each historical submission.
- This is a retrospective source and score audit, not a fresh end-to-end reproduction. No new translation quality, speed, cost, ablation gain or original-training claim is made.
- This writeup was assembled with AI assistance from the recovered artifacts. It is intended to make the evidence inspectable and to support a technical discussion of the actual limitations.


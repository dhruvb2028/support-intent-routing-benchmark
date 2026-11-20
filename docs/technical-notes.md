# Technical notes

These notes describe the portfolio's recorded evidence. They were prepared with AI assistance; model-training and data-review work also used AI agents. No company-directed outreach is included.

## 1. Apostrophes revealed leakage in the first data split

The support-intent benchmark starts with 26,872 hybrid synthetic requests from Bitext. Canonical deduplication removed 3,046 repeated requests. I then grouped conservative near duplicates before allocating training, validation and test data.

An independent AI reviewer found a flaw: punctuation replacement treated contractions and hyphens differently from punctuation removal. Variants such as `can't`/`cant` and `sign-up`/`signup` could still cross partitions. The reviewer found 30 cross-partition canonical collisions in the first split.

The corrected pipeline adds equivalence edges under both normalization rules. The resulting 23,826 requests form 22,445 groups, with 16,681 training, 3,575 validation and 3,570 test rows. Each intent appears in every partition. Identifiers, group metadata, response text, categories and generation flags are excluded from model features.

The practical lesson is to challenge the normalization rule itself, then verify separation independently. Pinned source hashes, source-row mappings and split manifests make that check repeatable. Surface grouping still cannot recover every original generation family, and synthetic test performance cannot establish real customer performance.

Evidence: [data report](data-report.md). Authorship: prepared with AI assistance; the independent review was performed by another AI agent, not human annotators.

## 2. A more elaborate pipeline needed a controlled comparison

My recovered Deep Past submission history contains an earlier private score of 35.7355 and a later TTA submission score of 34.0462, a difference of -1.6893 points. Those are historical Kaggle records, not new experimental results.

The recovered source includes normalization, multiple translation candidates and consensus-based selection. The later notebook adds prompt and input variants, along with a gap-count heuristic. Several settings changed together. The available evidence does not link every historical submission to exact recovered settings, so the score difference cannot isolate TTA's effect.

The next useful experiment would keep one checkpoint and fixed decoding as a baseline, change one factor at a time, and save per-example outputs and fallback counts. Silent generic translations after inference errors deserve explicit accounting. Agreement among candidates is also only a heuristic: similar translations can share the same mistake.

The checkpoints came from community sources, including Assia Ben and Mattia Angeli; pretrained weights and inherited code are not my original training work. The contribution of this retrospective is a documented score audit and a clearer experimental protocol.

Evidence: [retrospective](deep-past-retrospective.md) and its executable score-arithmetic notebook. Authorship: assembled with AI assistance from recovered records; no fresh translation reproduction or causal ablation claim is made.

## 3. Quality and latency depended on the evaluation set

On the grouped Bitext test, TF-IDF plus logistic regression reached 99.64% macro-F1; frozen MiniLM embeddings plus a fitted linear classifier reached 99.42%. On 180 newly AI-authored supported requests, their forced-choice accuracies were 58.89% and 84.44%. The source-test ranking alone gave an incomplete picture of transfer.

Recorded warm batch-one median latency was 1.95 ms for TF-IDF on CPU and 5.96 ms for MiniLM on GPU. Each timing used 100 sequential requests and five warmups, excluding model loading and HTTP. These are measurements on one machine with different execution devices, not architecture-only speed comparisons or production guarantees.

The fixed Qwen2.5-1.5B zero-shot, 27-way scoring setup reached 27.75% source-test macro-F1 at a 46.30 ms warm median. That negative result applies to this prompt and scoring method. It does not establish a general limit on LLM classification.

The report reconstructs metrics from saved predictions and actually retrains the selected TF-IDF pipeline with frozen settings. Its labels and acceptance decisions match the original source-test predictions exactly in the recorded environment. No threshold was retuned after seeing the final challenge.

Evidence: [executed benchmark notebook](../notebooks/support-routing-benchmark.ipynb), [final metrics](../results/benchmark.json). Authorship: experiment, review and writing used AI assistance; pretrained MiniLM and Qwen weights are upstream work.

## 4. Rejection rates needed a coverage denominator

At the development-selected threshold of 0.89, TF-IDF rejected all 30 near-domain unsupported requests and 29/30 distant unsupported requests. It also deferred 52/60 ambiguous requests. MiniLM deferred all 60 unsupported requests and 53/60 ambiguous requests.

The cost was substantial: TF-IDF accepted only 30/180 supported challenge requests, with 29 correct. MiniLM accepted 60/180, all correct in this sample. Accepted accuracy therefore came from small denominators and left most supported requests for human review. The approximate 95% interval for MiniLM's 60/60 accepted accuracy was 94.0–100%, rather than proof of perfect reliability.

The majority and Qwen thresholds deferred everything. Their 100% unsupported rejection rate illustrates why rejection without supported coverage is insufficient. Confidence scores were uncalibrated, and the threshold objective used a chosen balance of development costs rather than measured production prevalence.

Near-domain requests did not have worse rejection than distant requests here. A larger, human-reviewed, real-traffic evaluation would be needed before claiming reliable rejection or useful automation rates. The current result is a diagnostic finding, not deployment validation.

Evidence: [final challenge results](../results/benchmark.json), [data and review protocol](data-report.md). Authorship: the challenge was AI-authored and independently AI-reviewed; no human annotation study is claimed.

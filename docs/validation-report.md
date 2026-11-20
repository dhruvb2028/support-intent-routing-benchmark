# Independent technical validation

**Technical review: passed.** Core benchmark, dataset, executed benchmark notebook, retrospective notebook, demo and supplemental resource evidence passed the checks described below. The EDA update passed source/content checks; that check does not claim local execution of the old EDA.

Reviewer: independent Codex AI agent. Date: 21 November 2026 (Asia/Calcutta). This is an AI review, not human annotation.

## Verified result and its limits

The strongest finding is a gap between the grouped source test split and newly authored requests. High source-test performance does not establish production readiness.

| Model | Source test macro-F1 | Supported challenge accuracy before abstention | Supported challenge coverage | Warm median / p95, ms |
|---|---:|---:|---:|---:|
| Majority prior | 0.0030 | 3.89% | 0% | 0.33 / 0.62 |
| TF-IDF + logistic regression | 0.9964 | 58.89% | 16.67% | 1.95 / 2.76 |
| MiniLM + logistic regression | 0.9942 | 84.44% | 33.33% | 5.96 / 8.36 |
| Qwen2.5-1.5B, fixed zero-shot code prompt | 0.2775 | 35.56% | 0% | 46.30 / 50.43 |

The source test has 3,570 requests. The supported challenge has 180. TF-IDF correctly routes 29 of 30 accepted supported challenge requests; MiniLM correctly routes 60 of 60. These are small samples, and the associated intervals are recorded. TF-IDF accepts 9 of 120 ambiguous/unsupported requests; MiniLM accepts 7. Majority and Qwen defer everything at their selected thresholds, so selective accuracy is undefined, not 100%.

Latency is 100 sequential, batch-one requests after five warmups on the recorded local hardware. Timers include adapter preprocessing/tokenization, model forward, score conversion and GPU synchronization; they exclude model loading, HTTP, network and queueing. These values are not production service latency or measured cloud cost.

The fixed Qwen prompt performed poorly. Development-only sanity checks show that ordinary greedy generation emits the same incorrect codes on eight inspected examples. Batch versus single-request predictions agree for 27 examples, with disclosed FP16 score differences. This supports retaining the observed negative result; it does not establish performance of other prompts, fine-tuning methods, models or LLMs generally.

## Evidence and reproducibility

- `python -m pytest -q`: **34 passed**, including 30 independently written data, challenge, metric and freeze-barrier tests.
- A fresh isolated rerun of data preparation reproduced the exact training, validation and test hashes. Evidence: `results/review/data_reproduction.json`.
- `scripts/review_classical_reproduction.py` independently reconstructs both classical candidates and the threshold objective using development data only. Both candidate macro-F1 values match exactly; all 3,575 validation predictions agree; maximum absolute probability difference is 1.08e-14. The selected C=4 and threshold=0.89 reproduce. Evidence: `results/review/classical_reproduction.json`.
- `scripts/review_selection.py` reconstructs all four validation macro-F1 values and threshold utilities and checks data, source and artifact hashes. Evidence: `results/review/prefreeze_selection.json`.
- `scripts/review_final_outputs.py` reconstructs source accuracy, explicit-ontology macro-F1, confusion matrices, selective metrics, challenge slices, and latency summaries from saved predictions. All 15,480 prediction rows and 400 timing samples pass; IDs, probability arrays, configurations and hashes agree. Evidence: `results/review/final_outputs.json`.
- This review did not independently rerun all GPU inference. It inspected adapter sanity evidence and reconstructed recorded GPU outputs. The classical pipeline and data preparation were rerun independently.

The reviewed selections were frozen at 20:45:19.851799 UTC on 20 November 2026; final evaluation began at 20:45:23.717332 UTC, after the freeze. The independent pre-freeze audit precedes both, and its selected-record hashes are unchanged.

- Frozen selection SHA256: `5dd505704228aff925c118927ff43358180c61cfe6e3d8e575b5de94ee406c0c`.
- Final source test SHA256: `118eaee9a6485bbe9bff5240b3fb8060b29dc82cf01e13794589a4997f1b5f6e`.
- Final challenge SHA256: `fd726c6f18e0a45bf581bd6aa9e533d516072367300f1a6930119bda1e054331`.

Selection reads only training, validation and challenge development. Final evaluation checks frozen source, model, selection and final-data hashes before parsing final examples. Independent mutation tests confirm that changed inputs stop execution before final CSV parsing. Two explanatory ablations, word-only features and 10% training data, use validation only. The demo's TF-IDF choice precedes final challenge results; a new model choice based on those results would need to be described as post-evaluation selection.

## Findings repaired before final evaluation

| Finding | Verified resolution |
|---|---|
| Punctuation-equivalent requests crossed partitions: 20 train/validation, 7 train/test, 3 validation/test | Added punctuation-elided equivalence edges, rebuilt splits and invalidated preliminary development runs. Both normalization forms now pass. |
| Malformed unsafe-development scores and unknown challenge kinds/labels were accepted | Explicit input validation added; regression checks pass. |
| Final files were recorded only after evaluation, and source could change between selection and freeze | Final hashes pinned at freeze, source hashes recorded at selection, changed/stale artifacts rejected. Seven freeze mutation tests pass. |
| Instruction code validation did not check the actual assistant boundary, and whole-prompt truncation could remove that boundary | Boundary token checks added; oversized prompts are rejected without truncation. |
| Initial out-of-domain challenge used only distant topics | Replaced 30 unreviewed final requests with near-domain unsupported requests before selection; added development counterparts and separate subtype metrics. |
| Source preparation overwrote the detailed ontology, and cached raw bytes lacked expected-hash verification | Ontology preserved; cached and downloaded source/card bytes checked against pinned hashes. |
| Dataset upload omitted review-input files referenced by its manifest | Both review inputs included; exact 22-file allowlist and all hash references verified. |

The final corpus retains 23,826 requests in 22,445 groups: 16,681 train, 3,575 validation, 3,570 test. All 27 intents appear in each partition. Only request text enters models; responses, categories, generation flags, labels and identifiers do not enter features. Exact/group separation is verified. Original synthetic generation-family IDs are unavailable, so this is a surface-similarity grouping heuristic, not proof of template independence.

## Challenge annotation review

The final challenge contains 180 supported requests, 60 ambiguous requests, 30 near-domain unsupported requests and 30 distant unsupported requests. Development has a separate 60 requests. No exact normalized challenge requests duplicate development or corpus requests.

The reviewer labeled 60 original requests using only IDs, text and ontology before author labels were released. Kind, label and expected-action agreement was 60/60. The frozen reviewer labels have SHA256 `1b68ea29ee62a391e9f9b70b43bff3cbbb421ed99bb09f84e277e27f77a94d24`. These 60 texts remain unchanged in the final challenge.

A further 30 replacement requests received label-hidden review, with 30/30 agreement. The reviewer was informed of the intended near-domain slice, so this supplement was not fully blind to kind. Both reviews include text-specific reasoning, timestamps and hashes under `data/review/`. Agreement between AI agents does not establish human annotation validity; correlated errors remain possible.

## Publication and historical-claim checks

The dataset upload has an exact 22-file allowlist, with 19 byte-identical copied assets, source attribution, a modification notice, CDLA-Sharing-1.0 license, and all declared challenge inputs. No private account files, model artifacts or credentials are present. Evidence: `results/review/dataset_package.json`.

The Deep Past retrospective's six scored historical submissions match the account export. The private-score difference 35.7355 minus 34.0462 is 1.6893. All three private evidence hashes match. The recovered source supports the stated two checkpoints, eight candidate pools, consensus/length ranking, gap heuristic, sampling-off branch above 500 rows, and generic error fallback. Both recovered notebooks have no saved outputs. The public narrative acknowledges community weights/inherited code and does not claim fresh training, isolated TTA gains, or verified historical source-to-submission linkage.

The public retrospective notebook contains two executed arithmetic cells with correct outputs, a matching narrative hash, and no private source attachments/paths. Its two-file publication package passed review. Evidence: `results/review/retrospective_package.json`.

The pinned cached cards and official model cards declare Apache-2.0 for [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Pretrained weights are external work. Bitext's [source dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) declares CDLA-Sharing-1.0; its data license remains distinct from project software licensing.


## Final notebook, demo and resource checks

The benchmark notebook has 17 cells, including seven executed code cells with saved outputs. Its compressed evidence payload exactly matches current benchmark, frozen selection, resource, prediction and failure-case artifacts. The train-only TF-IDF replay uses C=4 and threshold=0.89 and records 100% label and acceptance agreement on the source test under scikit-learn 1.7.2. Neural inference and latency are clearly described as recorded results, not newly executed in that notebook. The quality/latency figure was visually checked. Evidence: `results/review/benchmark_notebook.json`.

The six-file demo package includes the MIT code license and AI-assistance disclosure. Its model hash, class labels and threshold match the frozen selection. Empty and oversized requests are handled, a supported request routes, and the tested ambiguous/unsupported requests are flagged. The UI explains 29/30 correct supported acceptances, nine unsafe acceptances out of 120, and 29/39 correct acceptances overall (74.4%). It does not present conditional accuracy as overall reliability or claim to perform support actions. Evidence: `results/review/demo_and_resources.json`.

The resource probe's script/data/frozen hashes and timing/memory arithmetic agree. It uses one fresh process and one input per model, cached assets, and potentially warm OS caches. Common benchmark imports include PyTorch even for CPU models; total sampled RSS is therefore benchmark-process memory, not minimal deployment memory. Baseline RSS and incremental peak are reported separately. GPU values are framework allocator peaks. No measured financial-cost or capacity claim follows from this probe.

The EDA update preserves 21 of 22 existing code cells and changes the dataset loader. All code cells compile; attribution, synthetic-data, descriptive-only and AI-assistance limits are present. Its original metadata identifies an already-public notebook. No local execution is claimed. Evidence: `results/review/eda_package.json`.

## Auditing a public clone

The public source repository omits trained model binaries and downloaded raw/weight caches. To reconstruct recorded metrics without asserting absent model hashes, run:

```bash
python scripts/review_final_outputs.py --allow-missing-model-artifacts
```

The report records `artifact_hash_verified: false` for each missing model. Without the explicit option, the audit requires model files and verifies their hashes. Source preparation must run before the full test suite because provenance tests verify raw source bytes. `scripts/start_reproduction.py` creates a new destination, copies verified source/data provenance, and executes the copied implementation; it refuses existing destinations and does not carry over old results or model artifacts. Because published outcomes are known, this is reproduction, not a new blind evaluation. Frozen byte hashes also require the repository's line-ending preservation rules.

## Reporting restrictions and remaining limits

Both the corpus and challenge are synthetic, English-only, narrow-domain data. Public data may have appeared in model pretraining. Bootstrap groups do not recover unknown generation families; challenge Wilson intervals assume independent examples. Small challenge slices and the declared 50/50 development utility do not represent production prevalence. Scores are uncalibrated.

Diagnostic failure counts combine raw classification errors on supported requests, including correctly deferred errors, with accepted ambiguous/unsupported requests. They must not be described as counts of wrong automatic actions. For example, TF-IDF's 95 diagnostic failures include 12 source classification errors, 74 supported-challenge classification errors, and 9 unsafe accepted requests; only one of the 74 supported-challenge errors was accepted.

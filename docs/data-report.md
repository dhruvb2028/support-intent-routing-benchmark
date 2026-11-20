# Data and evaluation provenance

The benchmark predicts one of 27 customer-support intents from a request alone. It uses Bitext's original hybrid synthetic dataset, plus a separately authored synthetic challenge set for routing and abstention. It does not establish performance on real customer traffic.

## Original source and attribution

Data provider: **Bitext Innovations**. Source: [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/tree/430d1a89bd93bd1fa23c16f29dd53e73f0087443), pinned revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`. The downloaded dataset card is retained in `data/raw/README.md`. It describes hybrid synthetic generation and declares **CDLA-Sharing-1.0**. That data license is separate from any software license in this repository. The [official license](https://cdla.dev/sharing-1-0/) is preserved in `data/LICENSE-CDLA-Sharing-1.0.html`.

Original CSV: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`.

- CSV SHA256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`.
- Dataset-card SHA256: `f9d490db6059f9749f443d47d3a38618dad168f61f439281738e29b396e451c2`.
- Source size: 26,872 rows, 27 intent labels, 11 observed category values. The card's prose says 10 categories; the downloaded bytes contain 11. Categories are unused features.

`fetch_source` verifies the expected hashes of both cached and newly downloaded files before processing. An edited cache fails rather than silently acquiring the pinned source's provenance.

**Modification notice:** the prepared corpus files are derivatives of Bitext data. This project removes duplicates, normalizes text solely for grouping, retains the original request text in model input, discards prohibited feature columns, assigns group/example identifiers, and partitions rows. Retain this notice, source attribution, source card, and data license when redistributing the data. The project does not claim ownership of the original corpus.

## Feature contract and partitioning

All three corpus CSVs contain exactly `example_id,text,label,group_id`. Only `text` is a model input; `label` is the intent target. `response`, `category`, `flags`, `intent`, identifiers, row position, and grouping metadata must never be model features. The instruction field becomes `text`, and the intent field becomes `label`.

First, Unicode NFKC, case folding, placeholder normalization, punctuation-to-space conversion, and whitespace collapse define canonical request text. Same-label canonical duplicates retain the first original row. Conflicting canonical labels would be quarantined and recorded in `data/audit/conflicting_labels.json`; none were found. This removes 3,046 redundant rows, leaving 23,826.

Second, a second punctuation-elided canonical form links variants such as `can't`/`cant` and `sign-up`/`signup` (94 equivalence edges). Additional edges require both character word-boundary 3-5gram TF-IDF cosine similarity at least 0.94 and RapidFuzz character ratio at least 95 (1,359 edges; no cross-label edges). Connected components define 22,445 groups; the largest contains 8 rows. Groups are kept together across partitions. Group construction uses unlabeled request strings across the corpus solely to prevent split leakage. Model vectorizers and learned transforms must fit only on training rows.

The allocation greedily minimizes per-intent squared deviation from 70/15/15 targets, assigning the largest groups first with deterministic seed 20261121. Every label is represented in every split.

| Partition | Rows | SHA256 |
|---|---:|---|
| Training | 16,681 | `cdda5bdcbf7173ae252eb335cd7688144193fe5b7ee468173e1125a3abc77883` |
| Validation | 3,575 | `32dbd11d6f896558cb23007ffd7435bd2815d8d5a16d4590165cdeb4680e430b` |
| Final test | 3,570 | `118eaee9a6485bbe9bff5240b3fb8060b29dc82cf01e13794589a4997f1b5f6e` |

The machine-readable specification is `data/manifest.json`; original zero-based row mappings are in `data/audit/source_row_map.csv`. Rebuild using `python scripts/prepare_data.py` from the project directory with the locked environment.

## Review-driven corrections before evaluation

The independent reviewer identified 30 cross-partition punctuation-elided collisions in v1. These included contractions and joined hyphenated words. Version 2 adds those equivalence edges and passes both canonicalization checks. Preliminary v1 model development was invalidated; the ML owner reported no frozen selection or final-test evaluation before the hold. Model selection must use the regenerated v2 data.

No original generation-family identifiers are available. Conservative surface-similarity groups cannot prove semantic or template independence, so even the improved split can be optimistic for deployment. Public source material can also have appeared in a pretrained model's training data.

## Separate synthetic challenge

The data-authoring agent authored the text without viewing model predictions or evaluation outcomes. Authorship is **AI-assisted**, not human-collected customer data. The deterministic authoring program is `data/challenge_authoring.py`. The ontology and behavior rules are in `data/ontology.json`.

| Slice | Final challenge | Development challenge | Required behavior |
|---|---:|---:|---|
| Supported, single intent | 180 | 27 | Route to the correct intent |
| Ambiguous or multiple intents | 60 | 16 | Clarify or abstain |
| Unsupported, near domain | 30 | 9 | Abstain |
| Unsupported, far domain | 30 | 8 | Abstain |
| Total | 300 | 60 | |

The final challenge is `data/processed/challenge_test.csv`; the separate development challenge is `data/processed/challenge_development.csv`. The latter is available for threshold selection. The final 300 examples must not drive threshold choice, model choice, feature engineering, or repeated fixes after outcomes are observed. Corpus validation and challenge development results determine the frozen selection protocol.

Near-domain unsupported requests cover product specifications and compatibility, manuals and maintenance, stock availability, warranty repairs, and loyalty-point programs. These subjects fall outside the 27-intent routing ontology, even though they concern the same store. Far-domain requests include general knowledge and creative writing. Both require separate rejection metrics: the easier distant requests must not conceal poor near-domain behavior.

The final set originally contained only distant unsupported requests. Independent review identified that coverage weakness before final model selection. Version 2 replaces 30 unreviewed distant requests with near-domain requests and updates 9 development examples; all 60 initially reviewed texts are preserved. Replacement IDs and file hashes are in `data/challenge_manifest.json`. No model outcomes informed those changes.

Challenge columns are `example_id,text,label,kind,acceptable_labels,expected_action,author_provenance,ood_subtype`. `kind` is one of `supported`, `ambiguous`, `out_of_domain`. Unsupported and ambiguous gold labels are `__abstain__`; that marker is an evaluation target, never an extra training intent. `ood_subtype` is `near_domain`, `far_domain`, or empty for non-OOD rows. Gold action distinguishes clarification from abstention even if a classifier exposes a single abstain action. All annotation/provenance fields remain evaluation metadata.

## Independent annotation review and checks

A separate AI reviewer labeled 60 initial requests (20 per kind), receiving only request ID, text, and ontology. Their labels were saved before release of the author key; kind, label, and action agreement was 60/60. A further 30 near-domain replacements received label-hidden independent AI review; the reviewer was informed of the intended slice, so this is not fully blind to kind. The v2 comparison agrees on all 90 reviewed examples (60/60 original and 30/30 supplemental) for kind, label, and action. Original and supplemental records, audit hashes, and agreement reports are retained under `data/review/`, including `independent_agreement_v2.json`. This is not a human annotation or inter-rater study, and correlated AI errors remain possible.

The independent corpus suite verifies exact schema, global unique identifiers, disjoint groups, punctuation-elided request separation, all 27 labels in each partition, and absence of conflicting canonical labels. The challenge suite verifies composition, label/action rules, no duplicate challenge/development requests, no exact canonical source copies, and all declared hashes. Both six-test suites pass on v2; the full independent data/challenge/metrics suite passed all 23 checks. The data preparer also checks punctuation-to-space separation, so both normalization rules are enforced.

Limitations: English only; narrow intent taxonomy; synthetic author/style effects; small per-intent challenge counts; development has only 16 ambiguous and 17 unsupported examples; no real-world drift or production prevalence estimates; conservative grouping rather than recovered source-generation families. Results should be described as a reproducible portfolio benchmark and diagnostic evidence, not customer-service deployment readiness.

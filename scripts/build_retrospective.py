"""Build and execute a public, historical-score retrospective without model inference.

Run from any directory: python scripts/build_retrospective.py
The public Markdown is the narrative source. Recorded score rows are parsed from its
Markdown table. Private files are optional local verification inputs; their contents
are never embedded or copied into the public notebook or publication directory.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import shutil
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/deep-past-retrospective.md'
NOTEBOOK = ROOT / 'notebooks/deep-past-retrospective.ipynb'
STAGING = ROOT / 'publish/retrospective'
EVIDENCE = [
    ('Recovered earlier notebook', '4909d1aec2be0787986ce5cfcbd36461dcffbb5f4cd697c6e5dbaa38aa922948', '_private/deep-past/best/deeppastchallenge.ipynb'),
    ('Recovered TTA notebook', '45814148d2713c8f41cd35fea7017186853d845125fdfced6368407dfc910ee1', '_private/deep-past/tta/tta-deeppastchallenge.ipynb'),
    ('Private submission-history export', '850f4bc1ef31939f242ec198caf33a2f0bfa9f6d65e9940466c10e6ba65306fc', '_private/deep-past-submissions.json'),
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def table_rows(markdown):
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('| Submitted (Kaggle timestamp)'))
    end = start + 2
    records = []
    while end < len(lines) and lines[end].startswith('|'):
        parts = [part.strip() for part in lines[end].strip('|').split('|')]
        if len(parts) != 4:
            raise ValueError('Historical score table must contain four columns')
        date, description, public, private = parts
        for score in [public, private]:
            if not re.fullmatch(r'\d+\.\d{4}', score):
                raise ValueError('Unexpected or missing historical score; do not coerce blank values to zero')
        records.append({'date': date, 'description': description, 'public_score': public, 'private_score': private, 'submission_count': 2 if 'two submissions' in description else 1})
        end += 1
    if len(records) != 5:
        raise ValueError('Unexpected historical score table; review narrative and computations')
    return '\n'.join(lines[:start]), '\n'.join(lines[end:]), records

def build():
    markdown = DOC.read_text(encoding='utf-8-sig')
    prefix, suffix, records = table_rows(markdown)
    # Verify available local evidence without depending on private files in a public clone.
    verified = []
    for label, expected_hash, relative in EVIDENCE:
        path = ROOT / relative
        if path.exists():
            if sha256(path) != expected_hash:
                raise ValueError(f'Recorded evidence hash mismatch: {label}')
            verified.append(label)
    rank_match = re.search(r'position of (\d+) out of ([\d,]+) teams', markdown)
    if rank_match is None:
        raise ValueError('No explicit historical rank found in public narrative')
    rank, teams = int(rank_match[1]), int(rank_match[2].replace(',', ''))
    setup = '''from decimal import Decimal
from IPython.display import display, Markdown

# Historical observations transcribed in the public retrospective.
# Running this cell does not train a model, rerun inference, or query Kaggle.
recorded_rows = ''' + repr(records) + '''

header = "| Submitted | Recorded description | Submissions represented | Public score | Private score |\\n|---|---|---:|---:|---:|"
body = [f"| {r['date']} | {r['description']} | {r['submission_count']} | {Decimal(r['public_score']):.4f} | {Decimal(r['private_score']):.4f} |" for r in recorded_rows]
display(Markdown(header + "\\n" + "\\n".join(body)))
print(f"Displayed {sum(r['submission_count'] for r in recorded_rows)} scored historical submissions in {len(recorded_rows)} rows. Blank-score submissions are omitted, not treated as zero.")
'''
    computation = '''earlier = next(r for r in recorded_rows if r['description'] == 'DeepPastChallenge, version 3')
tta = next(r for r in recorded_rows if r['description'] == 'TTA DeepPastChallenge, version 3')
private_difference = Decimal(tta['private_score']) - Decimal(earlier['private_score'])
public_difference = Decimal(tta['public_score']) - Decimal(earlier['public_score'])
rank, team_count = ''' + repr((rank, teams)) + '''
rank_fraction_pct = Decimal(rank) / Decimal(team_count) * 100
print(f"Recorded TTA minus earlier private score: {private_difference:+.4f} points")
print(f"Recorded TTA minus earlier public score:  {public_difference:+.4f} points")
print(f"Recorded final position: {rank:,}/{team_count:,} teams (rank / team count = {rank_fraction_pct:.2f}%)")
print("The export does not identify final-selected submissions. These differences are observational, not a TTA ablation.")
'''
    fingerprints = '''## Artifact fingerprints

These SHA-256 values identify the recovered artifacts used to prepare this retrospective. The source notebooks and account export remain private and are not attached here. A fingerprint does not prove which historical settings produced a leaderboard score.

| Evidence artifact | Recorded SHA-256 |
|---|---|
''' + '\n'.join(f'| {label} | `{digest}` |' for label, digest, _ in EVIDENCE) + '''

The public narrative used to build this notebook has SHA-256 `''' + sha256(DOC) + '''`. Only the table arithmetic is executed in this notebook. No translation model is loaded, no new benchmark is run, and no historical score is presented as a fresh reproduction.
'''
    prefix += '\n\nThe following executable cells display the recorded scores and recompute the arithmetic. Their saved outputs are fresh calculations over historical observations, not new translation results.\n'
    suffix = suffix.replace('with SHA-256 hashes in the evidence manifest', 'with SHA-256 hashes listed below')
    cells = [nbformat.v4.new_markdown_cell(prefix), nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(computation), nbformat.v4.new_markdown_cell(suffix), nbformat.v4.new_markdown_cell(fingerprints)]
    for i, cell in enumerate(cells):
        cell['id'] = hashlib.sha256(f'{i}\0{cell.source}'.encode()).hexdigest()[:12]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}, 'retrospective': {'kind': 'Historical source and recorded-score audit', 'source_document_sha256': sha256(DOC), 'model_inference_executed': False, 'training_executed': False, 'ai_assistance_disclosed': True}})
    NotebookClient(nb, timeout=120, kernel_name='python3', record_timing=False, resources={'metadata': {'path': str(ROOT)}}).execute()
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if cell.execution_count is None or not cell.outputs:
                raise AssertionError('Missing executed score audit output')
            if any(output.output_type == 'error' for output in cell.outputs):
                raise AssertionError('Notebook execution failed')
    nbformat.validate(nb)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)
    STAGING.mkdir(parents=True, exist_ok=True)
    allowed = {'deep-past-retrospective.ipynb', 'kernel-metadata.json'}
    unexpected = {path.name for path in STAGING.iterdir()} - allowed
    if unexpected:
        raise ValueError(f'Retrospective staging contains unexpected files: {sorted(unexpected)}')
    shutil.copyfile(NOTEBOOK, STAGING / NOTEBOOK.name)
    metadata = {'id': 'dhruvb2028/deep-past-score-audit-and-inference-lessons', 'title': 'Deep Past: Score Audit and Inference Lessons', 'code_file': NOTEBOOK.name, 'language': 'python', 'kernel_type': 'notebook', 'is_private': False, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': [], 'keywords': ['nlp']}
    (STAGING / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    assert sha256(NOTEBOOK) == sha256(STAGING / NOTEBOOK.name)
    public_bytes = NOTEBOOK.read_text(encoding='utf-8')
    assert '_private/' not in public_bytes and '_private\\' not in public_bytes
    print(json.dumps({'notebook': str(NOTEBOOK), 'staging': str(STAGING), 'executed_code_cells': 2, 'saved_outputs': True, 'private_artifacts_locally_hash_verified': len(verified), 'private_artifacts_copied': 0, 'notebook_sha256': sha256(NOTEBOOK), 'published': False}, indent=2))

if __name__ == '__main__':
    build()

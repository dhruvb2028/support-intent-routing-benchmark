"""Prepare an allowlisted Kaggle dataset upload; does not contact Kaggle."""
from pathlib import Path
import csv, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'publish/dataset'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def build():
    copies={
        **{f'{name}.csv':f'data/processed/{name}.csv' for name in ['train','validation','test','challenge_test','challenge_development']},
        'manifest.json':'data/manifest.json',
        'challenge_manifest.json':'data/challenge_manifest.json',
        'ontology.json':'data/ontology.json',
        'review_blind.csv':'data/review_blind.csv',
        'review_near_domain_blind.csv':'data/review_near_domain_blind.csv',
        'LICENSE-CDLA-Sharing-1.0.html':'data/LICENSE-CDLA-Sharing-1.0.html',
        'BITEXT_SOURCE_CARD.md':'data/raw/README.md',
        'DATA_REPORT.md':'docs/data-report.md',
        'source_row_map.csv':'data/audit/source_row_map.csv',
        'independent_agreement_v2.json':'data/review/independent_agreement_v2.json',
        'independent_blind_labels.csv':'data/review/independent_blind_labels.csv',
        'independent_blind_manifest.json':'data/review/independent_blind_manifest.json',
        'independent_supplement_labels.csv':'data/review/independent_supplement_labels.csv',
        'independent_supplement_manifest.json':'data/review/independent_supplement_manifest.json'
    }
    generated={'dataset-metadata.json','package-manifest.json','README.md'}
    DEST.mkdir(parents=True,exist_ok=True)
    unexpected={p.name for p in DEST.iterdir()}-set(copies)-generated
    if unexpected: raise ValueError(f'Upload directory contains non-allowlisted files: {sorted(unexpected)}')
    for dest,source in copies.items():
        shutil.copyfile(ROOT/source,DEST/dest)
        assert digest(ROOT/source)==digest(DEST/dest)
    description="""Audited customer-support intent-routing data with reproducible splits and a separate AI-authored abstention challenge.

**License: CDLA-Sharing-1.0.** Original corpus provider: Bitext Innovations. The source is the original Hugging Face dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset at revision 430d1a89bd93bd1fa23c16f29dd53e73f0087443. Source card and license are included. This upload does not replace the original data license with a software license.

**Modification notice:** Bitext requests were deduplicated, grouped by canonical and conservative near-duplicate equivalence, assigned identifiers, and partitioned. This package includes modified derivatives of Bitext data. Retain attribution and license when sharing.

**Files:** train.csv (16,681), validation.csv (3,575), test.csv (3,570); all 27 intents in each partition. Only text is a feature. Group IDs and labels are metadata/targets. No response, category or flags are included as model inputs. The retained corpus has 23,826 rows in 22,445 groups. Zero equivalent requests or groups cross splits under the two audited normalization rules.

**Challenge:** challenge_test.csv contains 300 newly AI-authored requests: 180 supported, 60 ambiguous, 30 near-domain unsupported and 30 far-domain unsupported. challenge_development.csv contains 60 separate development requests for threshold selection. Do not tune on the final challenge. An independent AI agent reviewed 90 final examples with agreement on labels/actions; this was not human annotation. Read DATA_REPORT.md for review design and limitations.

**Reproducibility:** manifest.json and challenge_manifest.json preserve original project-relative paths and hashes. package-manifest.json maps those source paths to this flat upload layout and verifies byte-identical copies. source_row_map.csv maps retained IDs to zero-based original CSV row positions. Train/validation/test were frozen before final evaluation.

**Limitations:** both source corpus and challenge are synthetic, English-only, narrow-domain data. Conservative near-duplicate grouping does not recover all original generation families. Results do not establish performance on real customer traffic.

Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/tree/430d1a89bd93bd1fa23c16f29dd53e73f0087443
License: https://cdla.dev/sharing-1-0/
"""
    # Kaggle's title limit is 50 characters; ampersand preserves the requested wording's meaning.
    title='Support Intent Routing: Audited Splits & Challenge'
    assert 6<=len(title)<=50
    meta={'title':title,'id':'dhruvb2028/support-intent-routing-audited-evaluation','subtitle':'27 intents, grouped splits, and a reviewed synthetic abstention challenge','description':description,'licenses':[{'name':'CDLA-Sharing-1.0'}],'keywords':['nlp','text classification'],'resources':[{'path':name,'description':('Frozen corpus split; only text is a feature.' if name in ['train.csv','validation.csv','test.csv'] else 'Synthetic diagnostic requests; metadata and labels must not be features.')} for name in ['train.csv','validation.csv','test.csv','challenge_test.csv','challenge_development.csv']]}
    (DEST/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    (DEST/'README.md').write_text('# '+title+'\n\n'+description+'\nSee DATA_REPORT.md for the full data card. The corpus preparation code is part of the companion portfolio repository; the manifests pin exact inputs and outputs.\n',encoding='utf-8')
    manifest={'prepared_on':'2026-11-21','purpose':'Kaggle upload staging only; not published by this script','license_slug':'CDLA-Sharing-1.0','license_support_source':'https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md#licenses','title_note':'Requested title exceeded 50 characters; replaced and with ampersand','source_revision':'430d1a89bd93bd1fa23c16f29dd53e73f0087443','copy_validation':'Every copied asset is byte-identical to its frozen project source','files':{name:{'project_source':source,'sha256':digest(DEST/name),'bytes':(DEST/name).stat().st_size} for name,source in copies.items()},'excluded':['Raw source request/response CSV (prepared derivatives included instead)','Model artifacts','Private career documents','Credentials','Environment/cache files']}
    (DEST/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for name,expected in [('train',16681),('validation',3575),('test',3570),('challenge_test',300),('challenge_development',60)]:
        with (DEST/f'{name}.csv').open(encoding='utf-8',newline='') as f: assert sum(1 for _ in csv.DictReader(f))==expected
    print(json.dumps({'directory':str(DEST),'files':len(list(DEST.iterdir())),'title_length':len(title),'license':meta['licenses'],'checks':'allowlist, source hashes, CSV row counts passed'},indent=2))
if __name__=='__main__': build()

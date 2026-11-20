"""Reproducible dataset provenance, duplicate audit, grouping, and split creation."""
from __future__ import annotations
import csv, hashlib, json, re, unicodedata, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_REPO = 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
SOURCE_REVISION = '430d1a89bd93bd1fa23c16f29dd53e73f0087443'
SOURCE_FILE = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
EXPECTED_SOURCE_SHA256 = '6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434'
EXPECTED_README_SHA256 = 'f9d490db6059f9749f443d47d3a38618dad168f61f439281738e29b396e451c2'
SEED = 20261121
FIELDS = ['example_id', 'text', 'label', 'group_id']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def canonical(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\{\{[^}]+\}\}', ' placeholder ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def canonical_elided(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\{\{[^}]+\}\}', ' placeholder ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def fetch_source(root):
    raw = Path(root) / 'data/raw'; raw.mkdir(parents=True, exist_ok=True)
    for filename in [SOURCE_FILE, 'README.md']:
        path = raw / filename
        if not path.exists():
            url = f'https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{filename}'
            with urllib.request.urlopen(url, timeout=180) as response:
                data = response.read()
            path.write_bytes(data)
        expected = EXPECTED_SOURCE_SHA256 if filename == SOURCE_FILE else EXPECTED_README_SHA256
        if sha256(path) != expected:
            raise ValueError(f'Source hash mismatch for {path}; refusing to attribute altered bytes to pinned revision')
    return raw / SOURCE_FILE

class UnionFind:
    def __init__(self, n): self.parent = list(range(n)); self.size = [1] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a == b: return
        if self.size[a] < self.size[b]: a,b = b,a
        self.parent[b] = a; self.size[a] += self.size[b]

def load_and_deduplicate(source):
    with Path(source).open(encoding='utf-8-sig', newline='') as f:
        source_rows = list(csv.DictReader(f))
    buckets = defaultdict(list)
    for idx, row in enumerate(source_rows):
        text, label = row['instruction'].strip(), row['intent'].strip()
        if not text or not label: raise ValueError(f'Empty input/label at source row {idx}')
        buckets[canonical(text)].append((idx, text, label))
    rows, conflicts = [], []
    duplicate_count = 0
    for norm, vals in buckets.items():
        labels = sorted(set(v[2] for v in vals))
        if len(labels) > 1:
            conflicts.append({'canonical_text':norm,'source_row_indices':[v[0] for v in vals],'labels':labels})
            continue
        idx,text,label = vals[0]
        duplicate_count += len(vals)-1
        example_id = 'bitext_' + hashlib.sha256(f'{idx}\0{text}\0{label}'.encode()).hexdigest()[:20]
        rows.append({'example_id':example_id,'text':text,'label':label,'canonical':norm,'source_row':idx})
    return rows, {'source_rows':len(source_rows),'source_columns':list(source_rows[0]),'source_label_counts':dict(Counter(r['intent'] for r in source_rows)),'exact_canonical_duplicates_removed':duplicate_count,'conflicting_canonical_texts':len(conflicts),'conflicting_source_rows_quarantined':sum(len(c['source_row_indices']) for c in conflicts)}, conflicts

def group_near_duplicates(rows):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rapidfuzz.fuzz import ratio
    texts = [r['canonical'] for r in rows]
    vectorizer = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float32,norm='l2')
    matrix = vectorizer.fit_transform(texts)
    uf = UnionFind(len(rows)); edge_count = 0; cross_label_edges = 0
    elided_first = {}; elided_edges = 0
    for i, row in enumerate(rows):
        key = canonical_elided(row['text'])
        if key in elided_first:
            j = elided_first[key]
            if rows[j]['label'] != row['label']:
                raise ValueError('Contradictory labels under punctuation-elided normalization; manual quarantine required')
            uf.union(i, j); elided_edges += 1
        else:
            elided_first[key] = i
    batch_size = 256
    for start in range(0,len(rows),batch_size):
        sims = (matrix[start:start+batch_size] @ matrix.T).tocoo()
        candidate = (sims.data >= 0.94) & ((sims.row + start) < sims.col)
        for left,right in zip(sims.row[candidate] + start,sims.col[candidate]):
            if ratio(texts[left],texts[right]) >= 95.0:
                uf.union(int(left),int(right)); edge_count += 1
                cross_label_edges += rows[left]['label'] != rows[right]['label']
        if start % 4096 == 0: print(f'Grouping {start}/{len(rows)} rows; {edge_count} edges',flush=True)
    groups = defaultdict(list)
    for idx in range(len(rows)): groups[uf.find(idx)].append(idx)
    for members in groups.values():
        digest = hashlib.sha256('\n'.join(sorted(rows[i]['example_id'] for i in members)).encode()).hexdigest()[:20]
        for idx in members: rows[idx]['group_id'] = 'group_' + digest
    return groups, {'punctuation_elided_equivalence_edges':elided_edges,'near_duplicate_edges':edge_count,'cross_label_near_duplicate_edges':int(cross_label_edges),'groups':len(groups),'largest_group_rows':max(map(len,groups.values())),'group_size_histogram':dict(sorted(Counter(map(len,groups.values())).items()))}

def allocate_splits(rows, groups):
    import numpy as np
    labels = sorted(set(r['label'] for r in rows)); li = {v:i for i,v in enumerate(labels)}
    fractions = np.array([0.70,0.15,0.15]); target = np.array([sum(r['label']==l for r in rows) for l in labels])[:,None] * fractions
    assignments = {}; counts = np.zeros_like(target)
    rng = np.random.default_rng(SEED)
    members_list = list(groups.values()); rng.shuffle(members_list)
    members_list.sort(key=len, reverse=True)
    # Place large groups first to minimize squared deviation from per-label row targets.
    for members in members_list:
        gc = np.zeros(len(labels))
        for idx in members: gc[li[rows[idx]['label']]] += 1
        costs=[]
        for split in range(3):
            before=((counts-target)**2 / np.maximum(target,1)).sum()
            trial=counts.copy(); trial[:,split]+=gc
            after=((trial-target)**2 / np.maximum(target,1)).sum()
            costs.append(after-before)
        chosen=int(np.argmin(costs)); counts[:,chosen]+=gc
        for idx in members: assignments[idx]=chosen
    names=['train','validation','test']
    splits={name:[] for name in names}
    for idx,row in enumerate(rows): splits[names[assignments[idx]]].append({key:row[key] for key in FIELDS})
    for split, vals in splits.items():
        vals.sort(key=lambda row:row['example_id'])
        missing=set(labels)-set(r['label'] for r in vals)
        if missing: raise ValueError(f'{split} missing labels: {missing}')
    return splits

def write_csv(path, rows, fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def prepare(root):
    root=Path(root); source=fetch_source(root)
    rows,audit,conflicts=load_and_deduplicate(source)
    print(json.dumps(audit),flush=True)
    groups,group_audit=group_near_duplicates(rows)
    splits=allocate_splits(rows,groups)
    processed=root/'data/processed';processed.mkdir(parents=True,exist_ok=True)
    for name,vals in splits.items(): write_csv(processed/f'{name}.csv',vals,FIELDS)
    write_json(root/'data/audit/conflicting_labels.json',conflicts)
    source_map=[{k:r[k] for k in ['example_id','source_row','group_id']} for r in rows]
    write_csv(root/'data/audit/source_row_map.csv',source_map,['example_id','source_row','group_id'])
    for a in splits:
        for b in splits:
            if a>=b: continue
            for field in ['example_id','group_id']:
                assert not ({r[field] for r in splits[a]} & {r[field] for r in splits[b]})
            assert not ({canonical(r['text']) for r in splits[a]} & {canonical(r['text']) for r in splits[b]})
            assert not ({canonical_elided(r['text']) for r in splits[a]} & {canonical_elided(r['text']) for r in splits[b]})
    manifest={'schema_version':2,'freeze_version':'v2-punctuation-elided-grouping','created_for':'Kaggle-to-YC portfolio benchmark','seed':SEED,'source':{'repository':SOURCE_REPO,'revision':SOURCE_REVISION,'filename':SOURCE_FILE,'expected_csv_sha256':EXPECTED_SOURCE_SHA256,'expected_readme_sha256':EXPECTED_README_SHA256,'hash_verification':'Downloaded and cached files are verified before parsing','csv_sha256':sha256(source),'readme_sha256':sha256(root/'data/raw/README.md'),'license':'CDLA-Sharing-1.0','url':f'https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}','nature':'Hybrid synthetic text from Bitext; not production customer traffic'},'input_columns':['text'],'target_column':'label','prohibited_model_inputs':['group_id','example_id','source_row','category','intent','response','flags'],'grouping':{'canonical':'Unicode NFKC; casefold; placeholder normalization; punctuation-to-space and whitespace normalization for deduplication; additional punctuation-elided equivalence edges before splitting','near_duplicate':'Connected components of character word-boundary 3-5gram TF-IDF cosine >=0.94 AND RapidFuzz ratio >=95.0','note':'Vocabulary is fit to all unlabeled inputs solely for data partitioning; downstream model transforms must fit on training only. Original generation family IDs are unavailable, so this is a conservative near-duplicate grouping heuristic, not proof of template independence.'},'audit':{**audit,**group_audit,'retained_rows':len(rows)},'split_method':'Deterministic group-wise greedy minimization of per-label squared row deviations from 70/15/15; largest groups first, seeded tie ordering','splits':{name:{'path':f'data/processed/{name}.csv','rows':len(vals),'groups':len(set(r['group_id'] for r in vals)),'label_counts':dict(sorted(Counter(r['label'] for r in vals).items())),'sha256':sha256(processed/f'{name}.csv')} for name,vals in splits.items()},'tests':{'example_id_disjoint':True,'group_id_disjoint':True,'canonical_text_disjoint':True,'punctuation_elided_text_disjoint':True,'all_labels_in_each_split':True},'limitations':['The source is synthetic and public; model pretraining contamination cannot be excluded.','Near-duplicate connected components do not recover all original generation families.','Independent challenge examples are also synthetic; real traffic validation remains necessary.']}
    write_json(root/'data/manifest.json',manifest)
    if not (root/'data/ontology.json').exists(): write_json(root/'data/ontology.json',{'labels':sorted(set(r['label'] for r in rows)),'task':'Return the supported intent for an unambiguous supported request; clarify or abstain for ambiguous/multiple intents; abstain for out-of-domain requests.'})
    print(json.dumps({'retained':len(rows),'groups':len(groups),'split_sizes':{k:len(v) for k,v in splits.items()},'manifest':str(root/'data/manifest.json')},indent=2),flush=True)
    return manifest

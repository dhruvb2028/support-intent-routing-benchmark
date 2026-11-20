"""Independent challenge/provenance release checks."""
from collections import Counter
from pathlib import Path
import csv, hashlib, json, re, unicodedata, unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    with (ROOT/'data/processed'/f'{name}.csv').open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def canonical(text):
    text = unicodedata.normalize('NFKC',text).casefold()
    text = re.sub(r'\{\{[^}]+\}\}', 'placeholder', text)
    return ' '.join(re.sub(r'[^\w\s]','',text).split())

class IndependentChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test = load('challenge_test')
        cls.dev = load('challenge_development')
        cls.labels = set(json.loads((ROOT/'data/ontology.json').read_text(encoding='utf-8'))['labels'])

    def test_final_composition_and_label_coverage(self):
        self.assertEqual(Counter(row['kind'] for row in self.test), {'supported':180,'ambiguous':60,'out_of_domain':60})
        self.assertEqual({r['label'] for r in self.test if r['kind']=='supported'},self.labels)

    def test_challenge_annotations_are_consistent(self):
        for row in self.test+self.dev:
            self.assertTrue(row['text'].strip())
            if row['kind']=='supported':
                self.assertIn(row['label'],self.labels)
                self.assertEqual(row['expected_action'],'route')
            else:
                self.assertIn(row['kind'],{'ambiguous','out_of_domain'})
                self.assertEqual(row['label'],'__abstain__')
                self.assertEqual(row['expected_action'],'clarify_or_abstain' if row['kind']=='ambiguous' else 'abstain')

    def test_challenge_ids_and_requests_do_not_repeat(self):
        rows = self.test+self.dev
        self.assertEqual(len({r['example_id'] for r in rows}),len(rows))
        self.assertEqual(len({canonical(r['text']) for r in rows}),len(rows))

    def test_challenge_requests_are_not_source_split_copies(self):
        source = {canonical(r['text']) for split in ('train','validation','test') for r in load(split)}
        challenge = {canonical(r['text']) for r in self.test+self.dev}
        self.assertFalse(source & challenge, 'Challenge request duplicates a source split request')

    def test_declared_source_and_split_hashes_match(self):
        manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
        for name,split in manifest['splits'].items():
            path=ROOT/split['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),split['sha256'])
            self.assertEqual(len(load(name)),split['rows'])
        raw = ROOT/'data/raw'/manifest['source']['filename']
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),manifest['source']['csv_sha256'])

    def test_declared_challenge_hashes_match(self):
        manifest=json.loads((ROOT/'data/challenge_manifest.json').read_text(encoding='utf-8'))
        for path,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'data'/path).read_bytes()).hexdigest(),digest,path)

if __name__=='__main__': unittest.main()

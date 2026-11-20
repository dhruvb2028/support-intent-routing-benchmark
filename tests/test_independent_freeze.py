"""Mutation checks for the final-evaluation freeze barrier, without training models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_benchmark_under_test',PROJECT/'scripts/run_benchmark.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class IndependentFreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.names=patch.object(runner,'MODEL_NAMES',['majority'])
        self.names.start()
        for source in runner.SOURCES:
            p=self.root/source;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen source fixture')
        for name in ['train','validation','challenge_development','test','challenge_test']:
            p=self.root/'data/processed'/f'{name}.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('example_id,text,label\n1,request,a\n')
        artifact=self.root/'artifacts/majority.joblib';artifact.parent.mkdir(parents=True);artifact.write_bytes(b'fixture - never deserialize')
        digest=lambda name:runner.sha(self.root/'data/processed'/f'{name}.csv')
        selection={'artifact':'artifacts/majority.joblib','artifact_sha256':runner.sha(artifact),
                   'data_sha256':{name:digest(name) for name in ['train','validation','challenge_development']},
                   'source_sha256':{p:runner.sha(self.root/p) for p in runner.SOURCES}}
        runner.write(self.root/'results/selection/majority/selection.json',selection)
        runner.write(self.root/'data/manifest.json',{'splits':{'test':{'sha256':digest('test')}}})
        runner.write(self.root/'data/challenge_manifest.json',{'files':{'processed/challenge_test.csv':digest('challenge_test')}})

    def tearDown(self):
        self.names.stop();self.tmp.cleanup()

    def assert_blocks_before_read(self):
        with patch.object(runner.pd,'read_csv') as read:
            with self.assertRaises(RuntimeError): runner.test(self.root,[],'cpu')
            read.assert_not_called()

    def test_final_data_is_hash_frozen(self):
        runner.freeze(self.root)
        value=json.loads((self.root/'results/frozen_selection.json').read_text())
        self.assertEqual(set(value['final_data_sha256']),{'test','challenge_test'})
        for name in ['test','challenge_test']:
            self.assertEqual(value['final_data_sha256'][name],runner.sha(self.root/'data/processed'/f'{name}.csv'))

    def test_modified_final_data_rejected_before_read(self):
        runner.freeze(self.root)
        (self.root/'data/processed/challenge_test.csv').write_text('modified final input')
        self.assert_blocks_before_read()

    def test_modified_source_rejected_before_read(self):
        runner.freeze(self.root)
        (self.root/runner.SOURCES[0]).write_text('modified implementation')
        self.assert_blocks_before_read()

    def test_modified_model_rejected_before_read(self):
        runner.freeze(self.root)
        (self.root/'artifacts/majority.joblib').write_bytes(b'modified model')
        self.assert_blocks_before_read()

    def test_modified_selection_rejected_before_read(self):
        runner.freeze(self.root)
        (self.root/'results/selection/majority/selection.json').write_text('{}')
        self.assert_blocks_before_read()

    def test_source_changed_since_selection_prevents_freeze(self):
        (self.root/runner.SOURCES[0]).write_text('source changed before freeze')
        with self.assertRaises(RuntimeError): runner.freeze(self.root)

    def test_freeze_cannot_be_overwritten(self):
        runner.freeze(self.root)
        before=(self.root/'results/frozen_selection.json').read_bytes()
        with self.assertRaises(RuntimeError): runner.freeze(self.root)
        self.assertEqual(before,(self.root/'results/frozen_selection.json').read_bytes())

if __name__=='__main__': unittest.main()

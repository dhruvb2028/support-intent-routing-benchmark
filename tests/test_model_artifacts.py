import hashlib,json
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from kaggle_yc.models import make_tfidf
from kaggle_yc.inference import Router

def fitted():
    texts=['cancel my order','cancel the order','please cancel order','where is my delivery','track the delivery','find my delivery']
    return make_tfidf().fit(texts,['cancel']*3+['track']*3)

def setup_artifact(tmp_path):
    artifact=tmp_path/'artifacts/tfidf.joblib';artifact.parent.mkdir()
    joblib.dump(fitted(),artifact)
    frozen={'models':{'tfidf':{'artifact':'artifacts/tfidf.joblib','artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'config':{'threshold':.9}}}}
    (tmp_path/'results').mkdir()
    manifest=tmp_path/'results/frozen_selection.json'
    manifest.write_text(json.dumps(frozen))
    manifest.with_suffix('.sha256').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest())
    return artifact

def test_saved_model_preserves_probabilities(tmp_path):
    model=fitted();path=tmp_path/'model.joblib';joblib.dump(model,path)
    loaded=joblib.load(path)
    np.testing.assert_allclose(model.predict_proba(['cancel this order','find delivery']),loaded.predict_proba(['cancel this order','find delivery']))
    assert list(loaded.classes_)==['cancel','track']

def test_demo_rejects_empty_and_overlength_before_model(tmp_path):
    setup_artifact(tmp_path);router=Router(tmp_path)
    with patch.object(router.model,'predict_proba',side_effect=AssertionError('must not call model')):
        assert router.route('  ')['action']=='human_review'
        assert router.route('a'*2001)['action']=='human_review'

def test_corrupt_artifact_rejected_before_deserialization(tmp_path):
    artifact=setup_artifact(tmp_path);artifact.write_bytes(b'corrupt')
    with patch('kaggle_yc.inference.joblib.load') as loader:
        with pytest.raises(ValueError,match='hash'): Router(tmp_path)
        loader.assert_not_called()

def test_demo_exposes_abstention_and_uncalibrated_scores(tmp_path):
    setup_artifact(tmp_path);router=Router(tmp_path)
    result=router.route('some unknown request')
    assert result['action']=='human_review'
    assert result['scores_are_uncalibrated'] is True
    assert set(result['top_candidates'])=={'cancel','track'}

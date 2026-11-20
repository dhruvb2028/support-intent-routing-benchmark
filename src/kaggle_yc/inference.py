"""Load trusted local artifacts and route a single support request."""
from __future__ import annotations
import json, os, hashlib
from pathlib import Path
import joblib
from kaggle_yc.evaluate import decisions

class Router:
    def __init__(self,root,model_name='tfidf',device='cpu'):
        self.root=Path(root)
        os.environ.setdefault('HF_HOME',str(self.root/'.cache/huggingface'))
        path=self.root/'results/frozen_selection.json'
        if not path.exists(): raise FileNotFoundError('Run benchmark selection and freeze before launching the demo')
        hash_path=path.with_suffix('.sha256')
        if not hash_path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=hash_path.read_text().strip():
            raise ValueError('Frozen manifest hash mismatch')
        frozen=json.loads(path.read_text())
        if model_name not in frozen['models']: raise ValueError('Unknown model')
        selection=frozen['models'][model_name]
        artifact=self.root/selection['artifact']
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=selection['artifact_sha256']:
            raise ValueError('Model artifact hash differs from frozen selection')
        self.model=joblib.load(artifact)
        if hasattr(self.model,'device'): self.model.device=device
        self.threshold=selection['config']['threshold']
        self.name=model_name

    def route(self,text):
        text=text.strip()
        if not text: return {'action':'human_review','suggested_intent':None,'confidence':None,'reason':'Empty request'}
        if len(text)>2000: return {'action':'human_review','suggested_intent':None,'confidence':None,'reason':'Request exceeds demo length limit'}
        probs=self.model.predict_proba([text])
        pred,conf,accept=decisions(probs,self.model.classes_,self.threshold)
        order=probs[0].argsort()[::-1][:3]
        return {'action':'route' if accept[0] else 'human_review','suggested_intent':str(pred[0]),
                'confidence':float(conf[0]),'threshold':self.threshold,
                'scores_are_uncalibrated':True,
                'top_candidates':{str(self.model.classes_[i]):float(probs[0,i]) for i in order},
                'reason':'Above development threshold' if accept[0] else 'Below development threshold'}

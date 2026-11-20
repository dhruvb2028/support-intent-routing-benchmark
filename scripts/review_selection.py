"""Independent validation-only selection evidence audit before final test release."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def macro_f1(y,pred,labels):
    terms=[]
    for label in labels:
        tp=np.sum((y==label)&(pred==label));fp=np.sum((y!=label)&(pred==label));fn=np.sum((y==label)&(pred!=label))
        terms.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(terms))
def main():
    val=pd.read_csv(ROOT/'data/processed/validation.csv');dev=pd.read_csv(ROOT/'data/processed/challenge_development.csv')
    reports={}
    for name in ['majority','tfidf','minilm','qwen']:
        folder=ROOT/'results/selection'/name;path=folder/'selection.json';s=json.loads(path.read_text())
        for source,digest in s['source_sha256'].items():assert sha(ROOT/source)==digest,(name,source)
        for split,digest in s['data_sha256'].items():assert sha(ROOT/'data/processed'/f'{split}.csv')==digest,(name,split)
        assert sha(ROOT/s['artifact'])==s['artifact_sha256'],name
        scores=np.load(folder/'validation_predictions.npz',allow_pickle=False)
        ds=np.load(folder/'challenge_development_predictions.npz',allow_pickle=False)
        labels=scores['labels'];p=scores['probabilities'];dp=ds['probabilities']
        assert np.array_equal(labels,ds['labels']) and np.array_equal(labels,s['config']['class_labels'])
        assert np.array_equal(scores['example_ids'],val.example_id.to_numpy(dtype=str))
        assert np.array_equal(ds['example_ids'],dev.example_id.to_numpy(dtype=str))
        assert p.shape==(len(val),len(labels)) and dp.shape==(len(dev),len(labels))
        for probs in [p,dp]:assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(axis=1),1,rtol=0,atol=1e-6)
        y=val.label.to_numpy();pred=labels[p.argmax(axis=1)];metric=macro_f1(y,pred,labels)
        assert np.isclose(metric,s['validation']['macro_f1'],rtol=0,atol=1e-12)
        correct=y==pred;conf=p.max(axis=1);unsafe=dp[dev.kind.to_numpy()!='supported'].max(axis=1);choices=[]
        for t in np.unique(np.r_[0.,np.linspace(.1,.99,90),1.000001]):
            accepted=conf>=t
            u=.5*(np.mean(correct&accepted)-3*np.mean((~correct)&accepted))+.5*np.mean(unsafe<t)
            choices.append((float(u),float(np.mean(accepted)),-float(t)))
        best=max(choices);threshold=-best[2]
        assert threshold==s['config']['threshold'],name
        assert np.isclose(best[0],s['threshold_selection']['utility'],rtol=0,atol=1e-12)
        reports[name]={'selection_sha256':sha(path),'validation_macro_f1':metric,'threshold':threshold,'threshold_utility':best[0],'reported_metrics_reconstructed':True,'data_source_artifact_hashes_verified':True}
    report={'reviewer':'independent Codex AI agent','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'All four validation/development selections; no final test inputs opened by this audit','script_sha256':sha(Path(__file__)),'models':reports,'review_passed':True}
    target=ROOT/'results/review/prefreeze_selection.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

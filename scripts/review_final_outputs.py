"""Independent reconstruction of final metrics from saved per-example evidence."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same(actual,expected,name):
    if expected is None: assert actual is None,name
    else: assert np.isclose(actual,expected,rtol=0,atol=1e-12),(name,actual,expected)
def basic(y,pred,accepted):
    correct=y==pred;n=len(y);na=int(accepted.sum())
    return {'count':n,'accepted':na,'coverage':na/n if n else None,
        'selective_accuracy':float(np.mean(correct[accepted])) if na else None,
        'selective_error':float(np.mean(~correct[accepted])) if na else None,
        'correctly_routed_fraction':float(np.mean(correct&accepted)) if n else None,
        'wrongly_routed_fraction':float(np.mean((~correct)&accepted)) if n else None}
def f1(y,pred,labels):
    result=[]
    for label in labels:
        tp=np.sum((y==label)&(pred==label));fp=np.sum((y!=label)&(pred==label));fn=np.sum((y==label)&(pred!=label))
        result.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return float(np.mean(result))

def main(allow_missing_model_artifacts=False):
    frozen_path=ROOT/'results/frozen_selection.json';frozen=json.loads(frozen_path.read_text());frozen_hash=sha(frozen_path)
    assert frozen_hash==(ROOT/'results/frozen_selection.sha256').read_text().strip()
    for name,digest in frozen['final_data_sha256'].items():assert sha(ROOT/'data/processed'/f'{name}.csv')==digest
    for name,digest in frozen['source_sha256'].items():assert sha(ROOT/name)==digest
    summary=json.loads((ROOT/'results/benchmark.json').read_text());assert summary['frozen_selection_sha256']==frozen_hash
    models={}
    for name,selection in frozen['models'].items():
        artifact=ROOT/selection['artifact']
        artifact_verified=artifact.is_file()
        if artifact_verified: assert sha(artifact)==selection['artifact_sha256']
        elif not allow_missing_model_artifacts: raise FileNotFoundError('Model artifact absent; reconstruct models or pass --allow-missing-model-artifacts for a recorded-output audit: '+str(artifact))
        assert sha(ROOT/'results/selection'/name/'selection.json')==frozen['selection_sha256'][name]
        folder=ROOT/'results/test'/name;metrics=json.loads((folder/'metrics.json').read_text());assert metrics==summary['models'][name]
        assert metrics['frozen_selection_sha256']==frozen_hash
        assert metrics['test_sha256']==frozen['final_data_sha256']['test']
        assert metrics['challenge_test_sha256']==frozen['final_data_sha256']['challenge_test']
        assert metrics['selected_config']==selection['config']
        for split in ['test','challenge_test']:
            truth=pd.read_csv(ROOT/'data/processed'/f'{split}.csv');saved=pd.read_csv(folder/f'{split}_predictions.csv')
            scores=np.load(folder/f'{split}_predictions.npz',allow_pickle=False);p=scores['probabilities'];labels=scores['labels']
            assert np.array_equal(truth.example_id.to_numpy(dtype=str),scores['example_ids'])
            assert np.array_equal(truth.example_id,saved.example_id)
            assert np.array_equal(truth.label,saved.label)
            assert np.array_equal(labels,np.asarray(selection['config']['class_labels']))
            assert p.shape==(len(truth),len(labels)) and np.isfinite(p).all() and (p>=0).all()
            assert np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-6)
            pred=labels[p.argmax(axis=1)];conf=p.max(axis=1);accepted=conf>=selection['config']['threshold']
            assert np.array_equal(saved.predicted_label,pred)
            assert np.allclose(saved.confidence,conf,rtol=0,atol=1e-12)
            assert np.array_equal(saved.accepted,accepted)
            result=metrics[split]
            if split=='test':
                y=truth.label.to_numpy()
                same(result['accuracy'],float(np.mean(y==pred)),name+' accuracy')
                same(result['macro_f1'],f1(y,pred,labels),name+' macro_f1')
                cm=np.zeros((len(labels),len(labels)),dtype=int);ix={v:i for i,v in enumerate(labels)}
                for a,b in zip(y,pred):cm[ix[a],ix[b]]+=1
                assert np.array_equal(cm,np.asarray(result['confusion_matrix']))
                for key,value in basic(y,pred,accepted).items():same(result['selective'][key],value,name+' '+key)
            else:
                mask=truth.kind.to_numpy()=='supported';unsafe=~mask
                same(result['unsafe_defer_rate'],float(np.mean(~accepted[unsafe])),name+' unsafe defer')
                for key,value in basic(truth.loc[mask,'label'].to_numpy(),pred[mask],accepted[mask]).items():same(result['supported'][key],value,name+' challenge '+key)
                same(result['supported']['accuracy_without_abstention'],float(np.mean(pred[mask]==truth.loc[mask,'label'].to_numpy())),name+' supported raw accuracy')
                for kind in ['supported','ambiguous','out_of_domain']:
                    subset=truth.kind.to_numpy()==kind
                    same(result[kind+'_defer_rate'],float(np.mean(~accepted[subset])),name+' '+kind+' defer')
                for subtype in ['near_domain','far_domain']:
                    subset=(truth.kind.to_numpy()=='out_of_domain')&(truth.ood_subtype.fillna('').to_numpy()==subtype)
                    sr=result['out_of_domain_by_subtype'][subtype]
                    same(sr['count'],int(subset.sum()),name+' '+subtype+' count')
                    same(sr['defer_rate'],float(np.mean(~accepted[subset])),name+' '+subtype+' defer')
        latency=metrics['latency'];raw=np.asarray(latency['raw_ms'])
        assert len(raw)==latency['sample_count']==100 and np.isfinite(raw).all() and (raw>0).all()
        assert latency['batch_size']==latency['concurrency']==1
        same(latency['median_ms'],float(np.median(raw)),name+' median latency')
        same(latency['p95_ms'],float(np.percentile(raw,95)),name+' p95 latency')
        models[name]={'rows_checked':{'test':len(pd.read_csv(ROOT/'data/processed/test.csv')),'challenge_test':300},'metrics_reconstructed':True,'artifact_hash_verified':artifact_verified,'latency_samples_checked':len(raw)}
    report={'reviewer':'independent Codex AI agent','created_utc':datetime.now(timezone.utc).isoformat(),'frozen_selection_sha256':frozen_hash,
        'script_sha256':sha(Path(__file__)),'models':models,'review_passed':True,
        'allow_missing_model_artifacts':allow_missing_model_artifacts,'model_artifacts_verified_count':sum(m['artifact_hash_verified'] for m in models.values()),
        'scope':'Reconstructs stored metric values and timing summaries. Model hashes are verified only for artifacts present locally; per-model flags record this. Does not independently rerun GPU model inference or establish production validity.'}
    target=ROOT/'results/review/final_outputs.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-missing-model-artifacts',action='store_true',help='Reconstruct recorded output metrics without requiring omitted trained model files; absence is explicitly recorded, not verified.')
    main(parser.parse_args().allow_missing_model_artifacts)

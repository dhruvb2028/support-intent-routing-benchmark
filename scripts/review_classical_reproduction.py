"""Independent reconstruction of the selected classical model using development data only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion,Pipeline
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def macro_f1(y,pred,labels):
    vals=[]
    for label in labels:
        tp=int(np.sum((y==label)&(pred==label)))
        fp=int(np.sum((y!=label)&(pred==label)))
        fn=int(np.sum((y==label)&(pred!=label)))
        vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(vals))

def main():
    selection_path=ROOT/'results/selection/tfidf/selection.json'
    selection=json.loads(selection_path.read_text());cfg=selection['config']
    data={name:pd.read_csv(ROOT/'data/processed'/f'{name}.csv') for name in ['train','validation','challenge_development']}
    train,val,dev=[data[n] for n in ['train','validation','challenge_development']]
    started=time.perf_counter();candidates=[];winner=None
    with threadpool_limits(limits=4):
        for candidate in selection['candidates']:
            c=candidate['C']
            model=Pipeline([('features',FeatureUnion([
                ('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_features=60000)),
                ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,min_df=2,max_features=100000))])),
                ('classifier',LogisticRegression(C=c,solver='lbfgs',max_iter=1500,random_state=42))])
            model.fit(train['text'],train['label'])
            p=model.predict_proba(val['text']);labels=model.classes_;pred=labels[p.argmax(axis=1)]
            metric=macro_f1(val['label'].to_numpy(),pred,labels)
            candidates.append({'C':c,'macro_f1':metric,'reported_macro_f1':candidate['validation_macro_f1']})
            if winner is None or metric>winner[0]:winner=(metric,c,model,p)
        metric,c,model,probs=winner
        dp=model.predict_proba(dev['text'])
    assert c==cfg['C'],'Independent validation selected another configuration'
    saved=np.load(ROOT/'results/selection/tfidf/validation_predictions.npz',allow_pickle=False)
    assert np.array_equal(model.classes_,saved['labels'])
    assert np.array_equal(val['example_id'].to_numpy(dtype=str),saved['example_ids'])
    maxdiff=float(np.max(np.abs(probs-saved['probabilities'])))
    same_predictions=float(np.mean(probs.argmax(axis=1)==saved['probabilities'].argmax(axis=1)))
    assert maxdiff<1e-6, f'Probability reproduction difference {maxdiff}'
    labels=model.classes_;correct=labels[probs.argmax(axis=1)]==val['label'].to_numpy();conf=probs.max(axis=1)
    unsafe_conf=dp[dev['kind'].to_numpy()!='supported'].max(axis=1)
    choices=[]
    for threshold in np.unique(np.r_[0.,np.linspace(.1,.99,90),1.000001]):
        accepted=conf>=threshold
        utility=.5*(np.mean(correct&accepted)-3*np.mean((~correct)&accepted))+.5*np.mean(unsafe_conf<threshold)
        choices.append((float(utility),float(np.mean(accepted)),-float(threshold)))
    best=max(choices);threshold=-best[2]
    assert threshold==cfg['threshold'],'Independent threshold differs'
    report={'reviewer':'independent Codex AI agent','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Reconstructed sklearn pipeline and threshold utility; train/validation/challenge_development only; no test inputs read',
        'script_sha256':sha(Path(__file__)),'selection_sha256_at_review':sha(selection_path),
        'input_sha256':{name:sha(ROOT/'data/processed'/f'{name}.csv') for name in data},
        'candidates':candidates,'selected_C':c,'selected_threshold':threshold,'threshold_utility':best[0],
        'maximum_absolute_probability_difference':maxdiff,'prediction_agreement':same_predictions,'probability_tolerance':1e-6,
        'validation_rows':len(val),'elapsed_seconds':time.perf_counter()-started,'thread_limit':4,
        'reproduction_passed':True}
    output=ROOT/'results/review/classical_reproduction.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

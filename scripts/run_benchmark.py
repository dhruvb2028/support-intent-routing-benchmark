"""Frozen-development benchmark. Never optimize after opening final test data."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.metadata, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0, str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from kaggle_yc.models import (make_majority, make_tfidf, EncoderClassifier, InstructionClassifier,
                              ENCODER_ID, LLM_ID, ENCODER_REVISION, LLM_REVISION, PROMPT_VERSION, time_requests)
from kaggle_yc.evaluate import classification_metrics, challenge_metrics, choose_threshold, decisions

MODEL_NAMES = ['majority','tfidf','minilm','qwen']
SEED = 42
SOURCES = ['src/kaggle_yc/models.py','src/kaggle_yc/evaluate.py','scripts/run_benchmark.py']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def utc(): return datetime.now(timezone.utc).isoformat()
def hardware():
    import torch
    return {'os':platform.platform(),'processor':platform.processor(),'python':platform.python_version(),
            'logical_cpu_count':os.cpu_count(),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),
            'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','transformers','sentence-transformers','huggingface-hub']}}
def resolve_revision(repo,expected):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo,revision=expected).sha

def frames(root):
    # These are the only examples available to the selection stage.
    base=root/'data/processed'
    return [pd.read_csv(base/f'{name}.csv') for name in ['train','validation','challenge_development']]

def predictions(path,frame,probabilities,labels,threshold):
    pred,conf,accepted=decisions(probabilities,labels,threshold)
    out=frame.drop(columns=['text']).copy()
    out['predicted_label']=pred; out['confidence']=conf; out['accepted']=accepted
    out.to_csv(path,index=False)
    np.savez_compressed(path.with_suffix('.npz'),probabilities=probabilities,labels=np.asarray(labels),example_ids=frame.example_id.to_numpy(dtype=str))

def score(y,probs,labels):
    return float(f1_score(y,np.asarray(labels)[np.argmax(probs,axis=1)],labels=labels,average='macro'))

def select(root,names,device):
    if (root/'results/frozen_selection.json').exists(): raise RuntimeError('Selection is frozen; use a new run directory for a genuinely new experiment')
    train,val,dev=frames(root)
    labels=sorted(train.label.unique())
    unsafe=dev.kind!='supported'
    data_hashes={name:sha(root/'data/processed'/f'{name}.csv') for name in ['train','validation','challenge_development']}
    for name in names:
        output=root/'results/selection'/name; output.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter(); candidates=[]
        print(f'{utc()} selecting {name}',flush=True)
        revision=None; config={'seed':SEED,'selection_metric':'validation_macro_f1','class_labels':labels}
        if name=='majority':
            model=make_majority().fit(train.text,train.label)
            vp=model.predict_proba(val.text); dp=model.predict_proba(dev.text)
            config.update({'method':'training_class_prior; argmax is majority label'})
        elif name=='tfidf':
            best=-1
            for c in [1.,4.]:
                candidate=make_tfidf(c,SEED).fit(train.text,train.label)
                probs=candidate.predict_proba(val.text)
                s=score(val.label,probs,candidate.classes_)
                candidates.append({'C':c,'validation_macro_f1':s})
                print(f'{utc()} tfidf C={c} val_macro_f1={s:.6f}',flush=True)
                if s>best:
                    model,vp,best=candidate,probs,s; config['C']=c
            dp=model.predict_proba(dev.text)
            config.update({'word_ngrams':[1,2],'character_ngrams':[3,5],'character_analyzer':'char_wb','min_df':2,
                           'max_word_features':60000,'max_char_features':100000,'solver':'lbfgs','max_iter':1500})
        elif name=='minilm':
            revision=resolve_revision(ENCODER_ID,ENCODER_REVISION)
            encoder=EncoderClassifier(revision,device=device)
            te=encoder.encode(train.text); ve=encoder.encode(val.text); de=encoder.encode(dev.text)
            best=-1
            for c in [1.,4.,16.]:
                candidate=EncoderClassifier(revision,c,SEED,device).fit_embeddings(te,train.label)
                probs=candidate.classifier.predict_proba(ve)
                s=score(val.label,probs,candidate.classes_)
                candidates.append({'C':c,'validation_macro_f1':s})
                print(f'{utc()} minilm C={c} val_macro_f1={s:.6f}',flush=True)
                if s>best: model,vp,best=candidate,probs,s; config['C']=c
            dp=model.classifier.predict_proba(de)
            model._encoder=encoder._encoder
            config.update({'model_id':ENCODER_ID,'revision':revision,'max_seq_length':256,'normalize_embeddings':True,
                           'encoder_finetuned':False,'solver':'lbfgs','max_iter':1500})
            del encoder,te,ve,de
        elif name=='qwen':
            revision=resolve_revision(LLM_ID,LLM_REVISION)
            model=InstructionClassifier(labels,revision,device=device)
            vp=model.predict_proba(val.text.tolist()); dp=model.predict_proba(dev.text.tolist())
            config.update({'model_id':LLM_ID,'revision':revision,'prompt_version':PROMPT_VERSION,
                           'prompt_template_messages':model.prompt('{request_text}'),'max_length':768,
                           'input_policy':'Reject rendered prompts exceeding 768 tokens; no truncation; fixed assistant suffix checked each call',
                           'truncated_requests':0,
                           'codes':model.codes[:len(labels)],'token_ids':model.token_ids,
                           'inference':'one-token constrained candidate logits; zero-shot; no finetuning',
                           'confidence':'uncalibrated softmax normalized ONLY across allowed class codes'})
        labels=list(model.classes_)
        if name=='tfidf':
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            word=Pipeline([('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_features=60000)),
                           ('classifier',LogisticRegression(C=config['C'],solver='lbfgs',max_iter=1500,random_state=SEED))]).fit(train.text,train.label)
            word_probs=word.predict_proba(val.text)
            small,_=train_test_split(train,train_size=.1,stratify=train.label,random_state=SEED)
            few=make_tfidf(config['C'],SEED).fit(small.text,small.label)
            few_probs=few.predict_proba(val.text)
            ablations={'scope':'validation only; explanatory analyses; not additional model-selection candidates',
                       'fixed_C':config['C'],'full_word_char_macro_f1':score(val.label,vp,labels),
                       'word_only_macro_f1':score(val.label,word_probs,word.classes_),
                       'ten_percent_training_macro_f1':score(val.label,few_probs,few.classes_),
                       'ten_percent_training_rows':len(small),'training_subset_seed':SEED}
            write(output/'ablations.json',ablations)
            predictions(output/'ablation_word_only_predictions.csv',val,word_probs,list(word.classes_),0.)
            predictions(output/'ablation_ten_percent_predictions.csv',val,few_probs,list(few.classes_),0.)
        threshold,curve=choose_threshold(val.label.to_numpy(),vp,labels,dp[unsafe])
        config['threshold']=threshold['threshold']
        artifact=root/'artifacts'/f'{name}.joblib'; artifact.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,artifact)
        predictions(output/'validation_predictions.csv',val,vp,labels,config['threshold'])
        predictions(output/'challenge_development_predictions.csv',dev,dp,labels,config['threshold'])
        write(output/'threshold_curve.json',curve)
        result={'name':name,'selected_at_utc':utc(),'config':config,'candidates':candidates,
                'threshold_selection':threshold,'threshold_objective':'0.5*(ID correct auto - 3*ID wrong auto) + 0.5*unsafe development defer; ties favor coverage',
                'threshold_assumption':'balanced ID/unsafe utility is a development choice, not an estimate of production prevalence',
                'validation':classification_metrics(val.label,vp,labels,config['threshold'],groups=val.group_id),
                'challenge_development':challenge_metrics(dev,dp,labels,config['threshold']),
                'elapsed_selection_seconds':time.perf_counter()-start,'device':device if name in ['minilm','qwen'] else 'cpu',
                'hardware':hardware(),'data_sha256':data_hashes,'artifact':str(artifact.relative_to(root)),
                'artifact_sha256':sha(artifact),'confidence_calibrated':False,
                'source_sha256':{p:sha(root/p) for p in SOURCES}}
        write(output/'selection.json',result)
        print(f'{utc()} selected {name}: F1={result["validation"]["macro_f1"]:.6f} threshold={config["threshold"]:.2f}',flush=True)
        if hasattr(model,'unload'): model.unload()
        del model; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def freeze(root):
    target=root/'results/frozen_selection.json'
    if target.exists(): raise RuntimeError('Already frozen')
    selections={name:json.loads((root/'results/selection'/name/'selection.json').read_text()) for name in MODEL_NAMES}
    data_hashes=next(iter(selections.values()))['data_sha256']
    for name,s in selections.items():
        if s['data_sha256']!=data_hashes: raise RuntimeError('Models were selected using different data')
        if sha(root/s['artifact'])!=s['artifact_sha256']: raise RuntimeError('Model artifact changed')
    for name,digest in data_hashes.items():
        if sha(root/'data/processed'/f'{name}.csv')!=digest: raise RuntimeError('Development data changed')
    for selection in selections.values():
        if selection['source_sha256']!={p:sha(root/p) for p in SOURCES}: raise RuntimeError('Source changed after selection')
    manifest=json.loads((root/'data/manifest.json').read_text())
    challenge_manifest=json.loads((root/'data/challenge_manifest.json').read_text())
    final_hashes={'test':manifest['splits']['test']['sha256'],
                  'challenge_test':challenge_manifest['files']['processed/challenge_test.csv']}
    for name,digest in final_hashes.items():
        if sha(root/'data/processed'/f'{name}.csv')!=digest: raise RuntimeError('Final data differs from data-author manifest')
    value={'frozen_at_utc':utc(),'models':selections,'selection_sha256':{n:sha(root/'results/selection'/n/'selection.json') for n in MODEL_NAMES},
           'source_sha256':{p:sha(root/p) for p in SOURCES},'final_data_sha256':final_hashes,'selection_policy':'All configurations and abstention thresholds frozen before reading test or challenge_test.',
           'final_evaluation_started':False}
    write(target,value)
    (root/'results/frozen_selection.sha256').write_text(sha(target)+'\n')
    print(f'Frozen manifest {sha(target)}',flush=True)

def test(root,names,device):
    target=root/'results/frozen_selection.json'
    frozen=json.loads(target.read_text())
    if sha(target)!=(root/'results/frozen_selection.sha256').read_text().strip(): raise RuntimeError('Frozen manifest hash mismatch')
    for p,digest in frozen['source_sha256'].items():
        if sha(root/p)!=digest: raise RuntimeError(f'Source changed after freeze: {p}')
    for name in MODEL_NAMES:
        selection=frozen['models'][name]
        if sha(root/selection['artifact'])!=selection['artifact_sha256']: raise RuntimeError('Artifact changed after selection')
        if sha(root/'results/selection'/name/'selection.json')!=frozen['selection_sha256'][name]: raise RuntimeError('Selection changed')
    base=root/'data/processed'
    for name,digest in frozen['final_data_sha256'].items():
        if sha(base/f'{name}.csv')!=digest: raise RuntimeError('Final data changed after freeze')
    marker=root/'results/final_evaluation_started.json'
    if not marker.exists(): write(marker,{'started_at_utc':utc(),'frozen_selection_sha256':sha(target)})
    # Final data is opened only below the verified freeze boundary.
    final=pd.read_csv(base/'test.csv'); challenge=pd.read_csv(base/'challenge_test.csv')
    for name in names:
        output=root/'results/test'/name
        if (output/'metrics.json').exists():
            existing=json.loads((output/'metrics.json').read_text())
            if existing['frozen_selection_sha256']!=sha(target) or existing['test_sha256']!=frozen['final_data_sha256']['test'] or existing['challenge_test_sha256']!=frozen['final_data_sha256']['challenge_test']:
                raise RuntimeError('Stale completed metrics; cannot combine different final evaluations')
            print(f'Skipping already complete final model {name}',flush=True); continue
        output.mkdir(parents=True,exist_ok=True)
        selected=frozen['models'][name]; threshold=selected['config']['threshold']
        print(f'{utc()} final evaluation {name}',flush=True)
        start=time.perf_counter(); model=joblib.load(root/selected['artifact'])
        if hasattr(model,'device'): model.device=device
        labels=list(model.classes_)
        probs=model.predict_proba(final.text.tolist()); cp=model.predict_proba(challenge.text.tolist())
        predictions(output/'test_predictions.csv',final,probs,labels,threshold)
        predictions(output/'challenge_test_predictions.csv',challenge,cp,labels,threshold)
        metrics={'name':name,'evaluated_at_utc':utc(),'selected_config':selected['config'],
                 'test':classification_metrics(final.label,probs,labels,threshold,groups=final.group_id),
                 'challenge_test':challenge_metrics(challenge,cp,labels,threshold),
                 'latency':time_requests(model,final.text,100),'hardware':hardware(),
                 'device':device if name in ['minilm','qwen'] else 'cpu',
                 'elapsed_evaluation_seconds':time.perf_counter()-start,
                 'test_sha256':sha(base/'test.csv'),'challenge_test_sha256':sha(base/'challenge_test.csv'),
                 'frozen_selection_sha256':sha(target)}
        # Fixed diagnostic ordering, declared before final evaluation: highest-confidence errors first.
        diagnostics=[]
        for split,frame,probabilities in [('test',final,probs),('challenge_test',challenge,cp)]:
            pr,co,ac=decisions(probabilities,labels,threshold)
            for i,row in enumerate(frame.to_dict('records')):
                supported=split=='test' or row['kind']=='supported'
                failure=(pr[i]!=row['label']) if supported else bool(ac[i])
                diagnostics.append({'split':split,**row,'predicted_label':pr[i],'confidence':float(co[i]),
                                    'accepted':bool(ac[i]),'failure':bool(failure),
                                    'failure_type':('classification_error' if supported else 'unsafe_request_accepted') if failure else 'correct_or_deferred'})
        diagnostic_frame=pd.DataFrame(diagnostics)
        errors=diagnostic_frame[diagnostic_frame.failure].sort_values('confidence',ascending=False,kind='stable')
        errors.head(20).to_csv(output/'failure_cases.csv',index=False)
        diagnostic_frame[~diagnostic_frame.failure].sort_values('confidence',kind='stable').head(20).to_csv(output/'other_diagnostics.csv',index=False)
        metrics['diagnostic_failure_count']=len(errors)
        metrics['failure_cases_exported']=min(20,len(errors))
        write(output/'metrics.json',metrics)
        print(f'{utc()} final {name}: F1={metrics["test"]["macro_f1"]:.6f}, latency={metrics["latency"]["median_ms"]:.2f}ms',flush=True)
        if hasattr(model,'unload'): model.unload()
        del model; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    paths=[root/'results/test'/name/'metrics.json' for name in MODEL_NAMES]
    if all(p.exists() for p in paths):
        write(root/'results/benchmark.json',{'frozen_selection_sha256':sha(target),'models':{n:json.loads(p.read_text()) for n,p in zip(MODEL_NAMES,paths)}})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['select','freeze','test'],required=True)
    parser.add_argument('--models',nargs='+',choices=MODEL_NAMES,default=MODEL_NAMES)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.stage=='select': select(args.root,args.models,device)
    elif args.stage=='freeze': freeze(args.root)
    else: test(args.root,args.models,device)
if __name__=='__main__': main()

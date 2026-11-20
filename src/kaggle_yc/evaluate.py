"""Evaluation with explicit selective prediction and challenge-set semantics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def checked_probabilities(probabilities, labels):
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(labels): raise ValueError("Probability shape mismatch")
    if not np.isfinite(probs).all() or (probs < 0).any(): raise ValueError("Invalid probabilities")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5): raise ValueError("Scores must sum to one")
    return probs

def decisions(probabilities, labels, threshold):
    probs = checked_probabilities(probabilities, labels)
    if not (0 <= threshold <= 1.000001): raise ValueError("Invalid threshold")
    idx = probs.argmax(axis=1)
    pred = np.asarray(labels)[idx]
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    return pred, conf, accepted

def wilson_interval(successes,total):
    if not total: return None
    z=1.959963984540054; p=successes/total; denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return {'low':max(0.,float(center-half)),'high':min(1.,float(center+half)),
            'method':'Wilson binomial interval; assumes independent examples','count':int(total)}

def selective_metrics(y, pred, accepted):
    y, pred, accepted = np.asarray(y), np.asarray(pred), np.asarray(accepted,dtype=bool)
    if not (len(y)==len(pred)==len(accepted)): raise ValueError("Length mismatch")
    n = len(y)
    n_accepted = int(accepted.sum())
    correct = y == pred
    return {"count":n,"accepted":n_accepted,"coverage":n_accepted/n if n else None,
            "selective_accuracy":float(correct[accepted].mean()) if n_accepted else None,
            "selective_error":float((~correct[accepted]).mean()) if n_accepted else None,
            "coverage_ci95":wilson_interval(n_accepted,n),
            "selective_accuracy_ci95":wilson_interval(int(correct[accepted].sum()),n_accepted),
            "correctly_routed_fraction":float((correct & accepted).mean()) if n else None,
            "wrongly_routed_fraction":float((~correct & accepted).mean()) if n else None}

def choose_threshold(y, probabilities, labels, unsafe_probabilities=None):
    """Predetermined balanced validation utility.

    0.5 * (correct automatic fraction - 3 * wrong automatic fraction)
    + 0.5 * defer fraction on development ambiguous/OOD requests.
    Values are not optimized on test examples. Ties prefer higher coverage.
    """
    pred, conf, _ = decisions(probabilities, labels, 0)
    unsafe_conf = checked_probabilities(unsafe_probabilities, labels).max(axis=1) if unsafe_probabilities is not None and len(unsafe_probabilities) else None
    thresholds = np.unique(np.r_[0., np.linspace(.1,.99,90), 1.000001])
    rows = []
    for t in thresholds:
        accepted = conf >= t
        sm = selective_metrics(y,pred,accepted)
        id_utility = sm['correctly_routed_fraction'] - 3*sm['wrongly_routed_fraction']
        defer = float((unsafe_conf < t).mean()) if unsafe_conf is not None else None
        utility = .5*id_utility+.5*defer if defer is not None else id_utility
        rows.append({"threshold":float(t),"utility":utility,"unsafe_defer_rate":defer,**sm})
    best = max(rows, key=lambda r:(r['utility'],r['coverage'],-r['threshold']))
    return best,rows

def bootstrap_accuracy(y,pred,seed=42,resamples=1000):
    correct = np.asarray(y)==np.asarray(pred)
    if not len(correct): return None
    rng=np.random.default_rng(seed)
    samples = [float(correct[rng.integers(0,len(correct),len(correct))].mean()) for _ in range(resamples)]
    return {"low":float(np.quantile(samples,.025)),"high":float(np.quantile(samples,.975)),"resamples":resamples,
            "method":"example bootstrap; correlated template groups may make this interval optimistic"}

def bootstrap_metric_intervals(y,pred,labels,accepted,groups=None,seed=42,resamples=1000):
    """Paired group bootstrap when group IDs exist, otherwise paired example bootstrap."""
    lookup={label:i for i,label in enumerate(labels)}
    truth=np.array([lookup[x] for x in y]); predicted=np.array([lookup[x] for x in pred])
    n=len(truth); k=len(labels)
    if n==0: return {}
    if groups is None: group_index=np.arange(n)
    else:
        if len(groups)!=n: raise ValueError('Group length mismatch')
        _,group_index=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    ng=int(group_index.max())+1; rng=np.random.default_rng(seed)
    values={name:[] for name in ['accuracy','macro_f1','coverage','selective_accuracy']}
    for _ in range(resamples):
        weights=np.bincount(rng.integers(0,ng,ng),minlength=ng)[group_index]
        cm=np.bincount(truth*k+predicted,weights=weights,minlength=k*k).reshape(k,k)
        denom=cm.sum(axis=0)+cm.sum(axis=1)
        f1=np.divide(2*np.diag(cm),denom,out=np.zeros(k),where=denom>0)
        total=weights.sum(); na=weights[accepted].sum()
        values['accuracy'].append(float(np.trace(cm)/total))
        values['macro_f1'].append(float(f1.mean()))
        values['coverage'].append(float(na/total))
        if na: values['selective_accuracy'].append(float(weights[accepted & (truth==predicted)].sum()/na))
    return {name:({'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),
                    'resamples':resamples,'seed':seed,'method':'paired group bootstrap' if groups is not None else 'paired example bootstrap'} if v else None)
            for name,v in values.items()}

def classification_metrics(y, probabilities, labels, threshold, groups=None):
    pred,conf,accepted=decisions(probabilities,labels,threshold)
    unknown=set(y)-set(labels)
    if unknown: raise ValueError(f"Unsupported ground truth labels: {unknown}")
    intervals=bootstrap_metric_intervals(y,pred,labels,accepted,groups)
    return {"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
            "accuracy_ci95":intervals["accuracy"],"macro_f1_ci95":intervals["macro_f1"],"intervals_ci95":intervals,
            "per_class":classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist(),"labels":list(labels),
            "selective":selective_metrics(y,pred,accepted)}

def challenge_metrics(frame, probabilities, labels, threshold):
    pred,conf,accepted=decisions(probabilities,labels,threshold)
    kind=frame['kind'].to_numpy()
    if len(frame) != len(pred): raise ValueError('Challenge length mismatch')
    if set(kind) - {'supported','ambiguous','out_of_domain'}: raise ValueError('Unknown challenge kind')
    supported=kind=='supported'
    if set(frame.loc[supported,'label']) - set(labels): raise ValueError('Unknown supported label')
    unsafe=~supported
    result={"count":len(frame),"supported_count":int(supported.sum()),"unsafe_count":int(unsafe.sum()),
            "unsafe_defer_rate":float((~accepted[unsafe]).mean()) if unsafe.any() else None,
            "unsafe_defer_rate_ci95":wilson_interval(int((~accepted[unsafe]).sum()),int(unsafe.sum())),
            "supported":selective_metrics(frame.loc[supported,'label'],pred[supported],accepted[supported])}
    result['supported']['accuracy_without_abstention']=float((frame.loc[supported,'label'].to_numpy()==pred[supported]).mean()) if supported.any() else None
    for subset in sorted(set(kind)):
        mask=kind==subset
        result[subset+'_defer_rate']=float((~accepted[mask]).mean())
        result[subset+'_defer_rate_ci95']=wilson_interval(int((~accepted[mask]).sum()),int(mask.sum()))
    if 'ood_subtype' in frame.columns:
        ood=kind=='out_of_domain'
        subtypes=frame['ood_subtype'].fillna('').to_numpy()
        if set(subtypes[ood]) - {'near_domain','far_domain'}: raise ValueError('Unknown OOD subtype')
        result['out_of_domain_by_subtype']={}
        for subtype in sorted(set(subtypes[ood])):
            mask=ood & (subtypes==subtype)
            result['out_of_domain_by_subtype'][subtype]={'count':int(mask.sum()),'defer_rate':float((~accepted[mask]).mean()),'defer_rate_ci95':wilson_interval(int((~accepted[mask]).sum()),int(mask.sum()))}
    return result

"""Supplemental operational measurements; no model or threshold selection.

One development request per model, fresh Python process. Local weights already
cached; OS cache may be warm. CUDA context is initialized before model timers.
Process-to-first-result includes interpreter/import/CUDA initialization time.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(ROOT/'src'))

def worker(name,device):
    import threading
    import psutil,torch
    from kaggle_yc.inference import Router
    torch.set_num_threads(4)
    if device=='cuda':
        torch.cuda.init();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    with (ROOT/'data/processed/validation.csv').open(encoding='utf-8',newline='') as handle: example=next(csv.DictReader(handle))
    process=psutil.Process();baseline=process.memory_info().rss;memory=[baseline];stop=threading.Event()
    def monitor():
        while not stop.wait(.01): memory.append(process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.perf_counter();router=Router(ROOT,name,device)
    if device=='cuda':torch.cuda.synchronize()
    load_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();result=router.route(example['text'])
    if device=='cuda':torch.cuda.synchronize()
    first_ready=time.perf_counter();first_ms=(first_ready-started)*1000
    started=time.perf_counter();router.route(example['text'])
    if device=='cuda':torch.cuda.synchronize()
    warm_ms=(time.perf_counter()-started)*1000
    memory.append(process.memory_info().rss);stop.set();thread.join()
    return {'model':name,'device':device,'validation_example_id':example['example_id'],'input_characters':len(example['text']),
            'artifact_deserialization_ms':load_ms,'first_request_ms_including_lazy_weights':first_ms,
            'model_load_plus_first_request_ms':load_ms+first_ms,'single_warm_repeat_ms':warm_ms,
            'fresh_process_to_first_result_ms':(first_ready-float(os.environ['PROBE_START_PERF']))*1000,
            'baseline_rss_mib':baseline/2**20,'peak_sampled_rss_mib':max(memory)/2**20,
            'peak_rss_increment_mib':(max(memory)-baseline)/2**20,'rss_sample_interval_ms':10,
            'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
            'probe_prediction':result,'torch':torch.__version__}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--device',default='cpu');args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker,args.device),allow_nan=False));return
    import torch
    accelerated='cuda' if torch.cuda.is_available() else 'cpu'
    results={}
    for name in ['majority','tfidf','minilm','qwen']:
        device=accelerated if name in ['minilm','qwen'] else 'cpu'
        env=dict(os.environ);env['PROBE_START_PERF']=str(time.perf_counter())
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name,'--device',device],env=env,capture_output=True,text=True,check=True)
        results[name]=json.loads(completed.stdout)
        print(f"{name}: first process result {results[name]['fresh_process_to_first_result_ms']:.1f}ms, RSS {results[name]['peak_sampled_rss_mib']:.1f}MiB",flush=True)
    output={'purpose':'Inference-only operational probe after selection freeze; no tuning',
            'frozen_selection_sha256':hashlib.sha256((ROOT/'results/frozen_selection.json').read_bytes()).hexdigest(),
            'validation_sha256':hashlib.sha256((ROOT/'data/processed/validation.csv').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'definitions':{'cold':'Fresh process and unloaded model; local model assets already cached, OS filesystem cache may be warm.',
                           'model_timers':'CUDA context initialized first; joblib load separate from first request which includes lazy encoder/LLM weight loading.',
                           'process_timer':'Interpreter startup, imports, GPU context, model load and first request to first result; excludes process shutdown.',
                           'memory':'PeakRSS sampled every10ms from baseline after common imports; may miss shorter peaks. CUDA values are PyTorch allocator peaks, not total physical GPU use.',
                           'limits':'One input and one cold run per model; descriptive operational probe, not a latency distribution or production capacity/cost estimate.'},
            'models':results}
    (ROOT/'results/resource_usage.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()

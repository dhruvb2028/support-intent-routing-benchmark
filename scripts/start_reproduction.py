"""Create a clean, fail-if-existing benchmark reproduction workspace."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def create_reproduction(destination,source=ROOT):
    source=Path(source).resolve()
    requested=Path(destination).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f'Destination already exists: {requested}')
    destination=requested.resolve()
    if destination==source or source in destination.parents:
        raise ValueError('Choose a new destination outside the published project directory')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Destination already exists: {destination}')
    files=[p.relative_to(source) for p in (source/'src').rglob('*.py')]
    files += [p.relative_to(source) for p in (source/'tests').glob('*.py')]
    files += [p.relative_to(source) for p in (source/'data/review').rglob('*') if p.is_file() and p.suffix in {'.json','.csv'}]
    files += [Path('scripts')/name for name in ['prepare_data.py','run_benchmark.py','measure_resources.py','start_reproduction.py']]
    files += [Path(name) for name in ['app.py','pyproject.toml','requirements-lock-windows-cu128.txt']]
    files += [Path('data')/name for name in ['manifest.json','challenge_manifest.json','ontology.json','README.md',
              'LICENSE-CDLA-Sharing-1.0.html','review_blind.csv','review_near_domain_blind.csv',
              'processed/challenge_development.csv','processed/challenge_test.csv']]
    for optional in ['LICENSE','.gitignore']:
        if (source/optional).is_file():files.append(Path(optional))
    files=sorted(set(files))
    # Validate all sources before creating anything; never follow source symlinks.
    for relative in files:
        path=source/relative
        if path.is_symlink() or not path.is_file() or source not in path.resolve().parents:
            raise ValueError(f'Missing or nonlocal source file: {relative}')
    destination.mkdir(parents=True,exist_ok=False)
    copied={}
    for relative in files:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,target)
        digest=sha(source/relative)
        if sha(target)!=digest:raise RuntimeError(f'Copy verification failed: {relative}')
        copied[relative.as_posix()]=digest
    manifest={'purpose':'Reproduce the published benchmark in a clean directory; this reuses known final data and is not a new blinded evaluation.',
              'copied_sha256':copied,'not_copied':['results','artifacts','.cache','.venv','data/raw','data/processed/train.csv','data/processed/validation.csv','data/processed/test.csv'],
              'expected_split_sha256':{n:s['sha256'] for n,s in json.loads((source/'data/manifest.json').read_text())['splits'].items()}}
    (destination/'reproduction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def run_pipeline(destination,device='auto'):
    destination=Path(destination).resolve();logdir=destination/'logs';logdir.mkdir(exist_ok=True)
    stages=[('prepare',['scripts/prepare_data.py']),
            ('select',['scripts/run_benchmark.py','--stage','select','--device',device]),
            ('freeze',['scripts/run_benchmark.py','--stage','freeze']),
            ('test',['scripts/run_benchmark.py','--stage','test','--device',device])]
    for name,arguments in stages:
        print(f'Running {name}; log: {logdir/(name+".log")}',flush=True)
        with (logdir/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,*arguments],cwd=destination,stdout=log,stderr=subprocess.STDOUT,check=True)
        if name=='prepare':
            expected=json.loads((destination/'reproduction_manifest.json').read_text())['expected_split_sha256']
            for split,digest in expected.items():
                if sha(destination/'data/processed'/f'{split}.csv')!=digest:
                    raise RuntimeError(f'Prepared split differs from published provenance: {split}')
    print(f'Complete: {destination/"results/benchmark.json"}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--run',action='store_true',help='Run prepare/select/freeze/test using this Python environment')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    manifest=create_reproduction(args.destination)
    print(f'Created {args.destination.resolve()} with {len(manifest["copied_sha256"])} hash-verified files',flush=True)
    if args.run:run_pipeline(args.destination,args.device)
    else:print('Preparation-only copy complete. Run the staged commands there using an environment with the pinned dependencies.',flush=True)
if __name__=='__main__':main()

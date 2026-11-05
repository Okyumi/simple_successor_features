"""Generate frozen experiment lists; launch one indexed run without shell eval."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def build_runs(stage='pilot', domains=None, seeds=None, coefficients=None, overrides=None):
    domains = domains or ['walker', 'cheetah']
    seeds = seeds if seeds is not None else ([0] if stage == 'smoke' else [0,1,2] if stage == 'pilot' else [10,11,12,13,14])
    coefficients = coefficients if coefficients is not None else ([0.,.1] if stage == 'smoke' else [0.,.01,.1,1.] if stage == 'pilot' else [0.,.1])
    if not seeds or not coefficients or any(s < 0 for s in seeds):
        raise ValueError('Provide nonnegative seeds and at least one coefficient')
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError('Coefficients must be finite and nonnegative')
    settings = dict(device='cuda', num_train_frames='2000000', num_seed_frames='4000',
        eval_every_frames='10000', num_eval_episodes='10', use_tb='true', use_wandb='false',
        **{'hydra.job.chdir':'true'}, save_snapshot_after_each_task='true',
        num_exposures='2' if stage=='transfer' else '1',
        terminate_after_first_task='false' if stage=='transfer' else 'true',
        reset_buffer_every_task='false', obs_type='pixels' if stage=='pixels' else 'states',
        action_repeat='2' if stage=='pixels' else '1')
    if stage == 'smoke':
        settings.update(num_train_frames='1200', num_seed_frames='1010',
            eval_every_frames='1000', num_eval_episodes='1', replay_buffer_num_workers='0',
            **{'agent.hidden_dim':'128','agent.sf_dim':'16','agent.batch_size':'16',
               'agent.geometry_every':'100'})
    for item in overrides or []:
        key, sep, value = item.partition('=')
        if not sep or key in ('agent','domain','seed','agent.nrc2_coef','hydra.run.dir') or key.startswith(('+','~')):
            raise ValueError(f'Invalid override {item!r}; use grid flags for domain/seed/coefficient')
        if ',' in value or '\n' in value:
            raise ValueError('Each --set is one Hydra value, not a multirun sweep')
        settings[key] = value
    runs=[]
    for domain,seed,coef in itertools.product(domains,seeds,coefficients):
        args=['agent=sf_nrc2',f'domain={domain}',f'seed={seed}',f'agent.nrc2_coef={coef}']
        args += [f'{k}={v}' for k,v in sorted(settings.items())]
        digest=hashlib.sha256(json.dumps(args).encode()).hexdigest()[:10]
        runs.append(dict(name=f'{stage}_{domain}_s{seed}_l{coef:g}_{digest}', overrides=args))
    if len({r['name'] for r in runs}) != len(runs):
        raise ValueError('Duplicate grid entries')
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','pilot','confirm','transfer','pixels'],default='pilot')
    p.add_argument('--domains',nargs='+',choices=['walker','cheetah'])
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--coefficients',nargs='+',type=float)
    p.add_argument('--set',dest='overrides',action='append',default=[],metavar='KEY=VALUE')
    p.add_argument('--write',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--total',action='store_true')
    p.add_argument('--index',type=int)
    p.add_argument('--run',action='store_true')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    a=p.parse_args()
    if a.manifest:
        if a.write or a.domains or a.seeds or a.coefficients or a.overrides:
            p.error('A frozen manifest cannot be combined with grid edits')
        data=json.loads(a.manifest.read_text()); runs=data['runs']
    else:
        runs=build_runs(a.stage,a.domains,a.seeds,a.coefficients,a.overrides)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        data=dict(version=1,git_commit=revision,runs=runs)
    if a.write:
        a.write.parent.mkdir(parents=True,exist_ok=True)
        with a.write.open('x') as f: json.dump(data,f,indent=2)
    if a.total:
        print(len(runs)); return
    if a.index is not None:
        if not 0 <= a.index < len(runs): p.error('Index outside manifest')
        runs=[runs[a.index]]
    if a.run and (not a.manifest or a.index is None):
        p.error('--run requires --manifest and --index (one job per experiment)')
    for r in runs:
        out=a.output_root.resolve()/r['name']
        cmd=[sys.executable,'-u',str(ROOT/'full_train.py'),*r['overrides'],f'hydra.run.dir={out}']
        print(shlex.join(cmd),flush=True)
        if a.run:
            # Atomic reservation prevents duplicate array tasks from mixing data.
            out.mkdir(parents=True,exist_ok=False)
            (out/'launch.json').write_text(json.dumps(dict(command=cmd,**data),indent=2))
            subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__': main()

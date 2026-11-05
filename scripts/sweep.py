"""Print reproducible commands; execute only with --run (large training budget)."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--run', action='store_true')
p.add_argument('--stage', choices=['pilot','confirm','transfer','pixels'], default='pilot')
p.add_argument('--device', default='cuda')
p.add_argument('--coefficient', type=float, default=.1)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
seeds = [0,1,2] if a.stage == 'pilot' else [10,11,12,13,14]
coefs = [0., .01, .1, 1.] if a.stage == 'pilot' else [0., a.coefficient]
for domain in ['walker','cheetah']:
    for seed in seeds:
        for coef in coefs:
            run_id = f'{a.stage}_{domain}_seed{seed}_lambda{coef:g}'
            cmd = [sys.executable, 'full_train.py', 'agent=sf_nrc2',
                   f'agent.nrc2_coef={coef}', f'domain={domain}', f'seed={seed}',
                   f'device={a.device}', 'num_train_frames=2000000',
                   'use_tb=true', 'use_wandb=false', 'hydra.job.chdir=true',
                   f'hydra.run.dir=outputs/{run_id}', 'save_snapshot_after_each_task=true']
            if a.stage == 'transfer':
                cmd += ['num_exposures=2', 'terminate_after_first_task=false',
                        'reset_buffer_every_task=false']
            else:
                cmd += ['num_exposures=1','terminate_after_first_task=true']
            if a.stage == 'pixels':
                cmd += ['obs_type=pixels', 'action_repeat=2']
            print(shlex.join(cmd), flush=True)
            if a.run:
                if (root/'outputs'/run_id).exists():
                    raise FileExistsError(f'Refusing to mix runs: {run_id}')
                subprocess.run(cmd, cwd=root, check=True, env=dict(os.environ,
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))

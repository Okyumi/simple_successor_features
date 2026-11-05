"""Seed-level return AUC and endpoint summaries from stationary eval logs."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
a=p.parse_args()
for run in a.runs:
    path=Path(run)/'full_train/eval.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if len({(r.get('task_id'),r.get('exposure_id')) for r in rows}) > 1:
        raise ValueError('Summarize each task/exposure separately for continual runs')
    x=np.array([r['frame'] for r in rows]); y=np.array([r['episode_reward'] for r in rows])
    if len(x)<2 or np.any(np.diff(x)<=0):
        raise ValueError('Need at least two strictly increasing evaluations')
    print(json.dumps(dict(run=run, evaluated_until=int(x[-1]),
          auc_return=float(np.trapezoid(y,x)/(x[-1]-x[0])),
          final_return=float(y[-1]), last5_mean=float(y[-5:].mean()))))

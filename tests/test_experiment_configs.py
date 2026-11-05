import json
import subprocess
import sys
import pytest
from experiment_configs import build_runs, ROOT


def test_grid():
    runs=build_runs()
    assert len(runs)==24 and len({r['name'] for r in runs})==24
    assert len(build_runs('smoke'))==4
    assert len(build_runs('confirm'))==20
    assert 'terminate_after_first_task=false' in build_runs('transfer')[0]['overrides']
    assert 'obs_type=pixels' in build_runs('pixels')[0]['overrides']


def test_overrides_and_validation():
    a=build_runs(overrides=['agent.lr=0.0003'])
    assert 'agent.lr=0.0003' in a[0]['overrides']
    assert a[0]['name']!=build_runs()[0]['name']
    for value in ['seed=9','hydra.run.dir=oops','agent.lr=1,2']:
        with pytest.raises(ValueError): build_runs(overrides=[value])
    with pytest.raises(ValueError): build_runs(coefficients=[float('nan')])
    with pytest.raises(ValueError): build_runs(seeds=[0,0])


def test_manifest_roundtrip(tmp_path):
    manifest=tmp_path/'grid.json'
    base=[sys.executable,str(ROOT/'experiment_configs.py')]
    subprocess.run(base+['--stage','smoke','--write',str(manifest),'--total'],check=True)
    assert len(json.loads(manifest.read_text())['runs'])==4
    r=subprocess.run(base+['--manifest',str(manifest),'--index','3'],capture_output=True,text=True,check=True)
    assert 'cheetah' in r.stdout and 'nrc2_coef=0.1' in r.stdout
    assert subprocess.run(base+['--manifest',str(manifest),'--index','4'],capture_output=True).returncode!=0

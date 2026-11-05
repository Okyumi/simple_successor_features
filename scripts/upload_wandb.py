"""Preview or upload saved JSONL metrics from a finished batch without retraining."""
import argparse
import hashlib
import json
from pathlib import Path


def read_run(run_dir):
    launch = json.loads((run_dir / 'launch.json').read_text())
    config = {'source': 'saved_jsonl', 'git_commit': launch['git_commit']}
    for arg in launch['command'][2:]:
        key, sep, value = arg.partition('=')
        if sep:
            try:
                value = json.loads(value)
            except ValueError:
                pass
            config[key.replace('.', '_')] = value
    records = []
    for prefix in ('train', 'eval'):
        path = run_dir / 'full_train' / (prefix + '.jsonl')
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                records.append((row['frame'], prefix, row))
    return config, sorted(records, key=lambda item: (item[0], item[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--project', default='nrc2_successor_features')
    parser.add_argument('--upload', action='store_true', help='Default is preview only')
    args = parser.parse_args()
    runs = sorted((args.batch / 'runs').glob('*/launch.json'))
    if not runs:
        parser.error('No run launch.json files found in this batch')
    for launch_path in runs:
        run_dir = launch_path.parent
        config, records = read_run(run_dir)
        identity = str(run_dir.resolve()) + args.entity + '/' + args.project
        run_id = 'import-' + hashlib.sha256(identity.encode()).hexdigest()[:20]
        marker = run_dir / (run_id + '.uploaded.json')
        if marker.exists():
            print('Already uploaded:', run_dir.name)
            continue
        print(run_dir.name, len(records), 'records', '->', args.entity + '/' + args.project)
        if not args.upload or not records:
            continue
        import wandb
        # Never silently append a second copy of an already-created history.
        with wandb.init(entity=args.entity, project=args.project, id=run_id,
                        resume='never', mode='online', name=run_dir.name,
                        group=config.get('experiment', run_dir.name.split('_')[0]),
                        config=config, dir=str(run_dir)) as run:
            for prefix in ('train', 'eval'):
                run.define_metric(prefix + '/frame')
                run.define_metric(prefix + '/*', step_metric=prefix + '/frame')
            for _, prefix, row in records:
                run.log({prefix + '/' + key: value for key, value in row.items()})
            url = run.url
        marker.write_text(json.dumps({'url': url, 'records': len(records)}, indent=2))
        print('Uploaded:', url)


if __name__ == '__main__':
    main()

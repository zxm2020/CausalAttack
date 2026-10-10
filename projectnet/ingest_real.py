"""Index synchronized REAL scene/texture/capture triples without changing the targets."""
import argparse
import csv
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='root/<session>/<sample>/{scene,texture,captured}.png')
    p.add_argument('--out', default='data/real_manifest.csv')
    args = p.parse_args()
    root = Path(args.input).resolve()
    sessions = [x for x in sorted(root.iterdir()) if x.is_dir()]
    if len(sessions) < 3:
        p.error('At least three physically distinct sessions are needed for train, val and test')
    rng = random.Random(2026)
    rng.shuffle(sessions)
    splits = {s.name: 'train' for s in sessions}
    splits[sessions[0].name] = 'test'
    splits[sessions[1].name] = 'val'
    rows = []
    for session in sessions:
        for sample in sorted(session.iterdir()):
            if not sample.is_dir():
                continue
            paths = [sample / f'{k}.png' for k in ('scene', 'texture', 'captured')]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            rows.append(dict(zip(('scene', 'texture', 'captured'), map(str, paths)), split=splits[session.name],
                             source='real_measured', scene_id=f'{session.name}/{sample.name}', session=session.name))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'texture', 'captured', 'split', 'source', 'scene_id', 'session'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Indexed {len(rows)} MEASURED real triples in {out}')


if __name__ == '__main__':
    main()

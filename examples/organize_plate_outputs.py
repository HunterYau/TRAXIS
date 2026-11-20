"""Group historical Wandowski runs; no simulation is launched. Preview by default."""
import argparse
import json
from pathlib import Path

from examples.plate_outputs import (PROJECT, RUNS_ROOT, CURRENT_RUN, LEGACY_RUNS, OLD_GROUPED_RUNS,
    read_saved, owner_alive, result_directory, trial_status, text_file)
from examples.pzt_terminal import export_queue_results
from traxis.storage import sha256_file, write_json


def inventory(root):
    files = [p for p in root.rglob('*') if p.is_file()]
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files),
                result_sha256={str(p.relative_to(root)):sha256_file(p)
                               for p in files if p.name in ('result.json', 'comparison.json', 'waveforms.npz')})


def move_inside_project(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    source.relative_to(PROJECT)
    target.relative_to(PROJECT)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def organize_current(root):
    """Flatten current frequency outputs and sort supporting files by purpose."""
    for trial in root.glob('*khz'):
        if not trial.is_dir():
            continue
        native = trial/'native'
        if native.is_dir():
            for entry in list(native.iterdir()):
                move_inside_project(entry, trial/entry.name)
            native.resolve().relative_to(PROJECT)
            native.rmdir()  # Empty after individually checked moves.
        if (trial/'summary.txt').exists():
            move_inside_project(trial/'summary.txt', root/'provenance/original-summaries'/f'{trial.name}.txt')
    reserved = {'prepared.json', 'queue-owner.json', 'all-results.json', 'all-results.txt',
                'queue-state.json', 'queue.lock', 'README.txt', 'STOP_AFTER_CURRENT'}
    for entry in list(root.iterdir()):
        if entry.name == 'cupy-cache':
            move_inside_project(entry, root/'cache/cupy')
        elif entry.is_file() and entry.name not in reserved:
            category = 'logs' if entry.suffix in ('.txt', '.log', '.jsonl') else 'provenance'
            move_inside_project(entry, root/category/entry.name)


def migrate():
    ledger_path = RUNS_ROOT/'relocations.json'
    ledger = read_saved(ledger_path) or {}
    old_group = PROJECT/'outputs/plate-experiments/wandowski-plate-1'
    def source_for(old_name):
        direct = PROJECT/'outputs'/old_name
        grouped = old_group/OLD_GROUPED_RUNS.get(old_name,old_name)
        return direct if direct.exists() else grouped
    # Refuse live/unknown ownership before any folder is moved.
    for old_name, new_name in LEGACY_RUNS.items():
        source = source_for(old_name)
        if source.exists():
            owner = read_saved(source/'queue-owner.json')
            if owner and owner_alive(owner) is not False:
                raise RuntimeError(f'Cannot move a live or unverified queue: {source}')
    for old_name, new_name in LEGACY_RUNS.items():
        source, target = source_for(old_name), RUNS_ROOT/new_name
        if source.exists():
            before = inventory(source)
            move_inside_project(source, target)
            after = inventory(target)
            if after != before:
                raise RuntimeError(f'Moved inventory differs: {target}')
            ledger[old_name] = dict(destination=str(target.relative_to(PROJECT)),
                                    moved_from=str(source.relative_to(PROJECT)),
                                    verified_before_reorganization=before)
            write_json(ledger_path, ledger, overwrite=True)
        if not target.exists():
            continue
        if target == CURRENT_RUN:
            if (target/'all-results.txt').exists() and not (target/'provenance/original-all-results.txt').exists():
                text_file(target/'provenance/original-all-results.txt', (target/'all-results.txt').read_text(encoding='utf-8'))
            organize_current(target)
        prepared = read_saved(target/'prepared.json')
        if prepared and 'configuration' in prepared:
            frequencies = prepared['configuration']['frequencies_hz']
            for frequency in frequencies:
                trial = target/f'{frequency//1000}khz'
                result = read_saved(result_directory(trial)/'result.json')
                if trial.exists():
                    status = 'completed' if result and result.get('completed') else 'interrupted'
                    if result and result.get('error'):
                        status = 'failed'
                    trial_status(trial,status,reason=(result or {}).get('error',
                        'Historical unfinished attempt; original files retained.' if status!='completed' else ''))
            export_queue_results(target,frequencies)
        text_file(target/'README.txt',
            f'Plate: Wandowski aluminum plate 1\nOriginal folder: {old_name}\n'
            f'Run: {new_name}\n'
            'See all-results.txt for results and completion status (if frequency tests were prepared).\n'
            'Setup-check folders contain short checks, not full experiments.\n'
            'Historical source snapshots and receipts are retained; old paths inside them are historical.\n'
            'Interrupted frequencies restart from the beginning; previous attempts remain saved.\n')
    rows = ['Simulation runs: each full experiment has its own folder', '',
            f'SELECTED: {CURRENT_RUN.name} (see its prepared receipt and all-results.txt).',
            'Latest completed realistic run: run-009-wandowski-realistic-bond-damping (50-250 kHz).',
            'Open its all-results.txt for all current results.', '',
            'Each run/preparation is a separate folder here. Short checks are in setup-checks/.',
            'Runs 006 and 007 together cover the earlier ideal-bond comparison; 007 continues 006.',
            'Different model versions remain separate experiments.', '', 'Old folder -> new location:']
    rows.extend(f'{old} -> {new}' for old,new in LEGACY_RUNS.items())
    text_file(RUNS_ROOT/'README.txt','\n'.join(rows)+'\n')
    text_file(PROJECT/'outputs/SIMULATION-RESULTS.txt',
              'Each full simulation run has its own folder under simulation-runs/.\n'
              f'Selected run results: simulation-runs/{CURRENT_RUN.name}/all-results.txt\n'
              'Historical path mapping and move verification: simulation-runs/relocations.json\n')
    # Preserve the earlier organization records, then remove only empty containers.
    previous = RUNS_ROOT/'provenance/previous-organization'
    for source in [old_group/'README.txt',old_group.parent/'relocations.json',
                   old_group.parent/'organization-verification.json',PROJECT/'outputs/PLATE-RESULTS.txt']:
        if source.exists():
            move_inside_project(source,previous/source.name)
    if old_group.exists():
        empty_candidates=sorted((p for p in old_group.rglob('*') if p.is_dir()),
                                key=lambda p:len(p.parts),reverse=True)
        for folder in [*empty_candidates,old_group,old_group.parent]:
            folder.resolve().relative_to(PROJECT)
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply:
        migrate()
        print(f'Organized simulation runs: {RUNS_ROOT}')
    else:
        for old,new in LEGACY_RUNS.items():
            print(f'outputs/{old} -> {RUNS_ROOT/new}')
        print('Preview only. Use --apply to move these folders inside the project.')


if __name__=='__main__':
    main()

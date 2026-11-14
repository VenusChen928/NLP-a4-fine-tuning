"""Build an immutable source release for upload to the course Burst project."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'part2').glob('*.py')) + [root / 'part2/requirements.txt']
    files += [root / 'hpc/burst' / name for name in (
        'part2.sbatch', 'part2_pipeline.sbatch', 'run_part2_gpu.sh', 'setup_part2_environment.sh',
        'package_part2.py', 'PART2.md')]
    sources = {str(path.relative_to(root)): digest(path) for path in files}
    data_files = sorted((root / 'part2/data').glob('*'))
    data_files.append(root / 'part2/records/ground_truth_dev.pkl')
    data = {str(path.relative_to(root / 'part2')): digest(path) for path in data_files if path.is_file()}
    code_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    release_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + code_hash[:12]
    destination = root / 'part2/releases' / release_id
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = {
        'release_id': release_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status_at_packaging': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'source_sha256': sources, 'data_sha256': data,
        'note': 'Source hashes identify the actual working-tree code, including uncommitted changes. Data are verified on Burst and are not included in this source archive.',
    }
    (destination / 'release_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = destination.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(destination, arcname=release_id)
    print(json.dumps({'release_id': release_id, 'directory': str(destination),
                     'archive': str(archive), 'archive_sha256': digest(archive)}, indent=2))


if __name__ == '__main__':
    main()

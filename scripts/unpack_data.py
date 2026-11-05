"""Expand benchmark inputs with path checks and SHA256 verification."""
from pathlib import Path
import tarfile, json, hashlib
root = Path(__file__).resolve().parents[1]
records = json.loads((root / 'data/files.json').read_text())
with tarfile.open(root / 'data/benchmark_inputs.tar.xz') as archive:
    for record in records:
        target = (root / record['path']).resolve()
        if root not in target.parents:
            raise ValueError('Invalid data path')
        content = archive.extractfile(record['path']).read()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Data hash mismatch: ' + record['path'])
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError('Refusing to replace changed input: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
print('Verified and expanded 18 benchmark input files.')

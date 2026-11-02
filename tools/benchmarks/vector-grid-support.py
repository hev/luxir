#!/usr/bin/env python3
"""Stage-1 grid helpers, isolated from the upstream comparison harness."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
import requests

spec = importlib.util.spec_from_file_location('vector_compare', Path(__file__).with_name('vector-compare.py'))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
DEFAULT_DATASET = compare.DEFAULT_DATASET
PRESETS = compare.PRESETS
LOOPBACK_BIND = compare.LOOPBACK_BIND
dataset = compare.dataset
checked = compare.checked


def download(url, path, sha=None, guard=None):
    if not path.exists():
        tmp = path.with_suffix('.partial')
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    if guard:
                        guard()
                    f.write(chunk)
        tmp.replace(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha and digest != sha:
        raise RuntimeError(f'Checksum mismatch: {path}')
    return digest


def validate_vectors(a, dims, allow_zero=False):
    if a.ndim != 2 or a.shape[1] != dims:
        raise RuntimeError('Wrong vector dimensions')
    if not np.isfinite(a).all():
        raise RuntimeError('Nonfinite vector')
    norms = np.linalg.norm(a, axis=1)
    if not allow_zero and np.any(norms == 0):
        raise RuntimeError('Zero vector')
    metric = 'ip' if np.allclose(norms, 1, rtol=0, atol=0.001) else 'cosine'
    return metric, {'min': float(norms.min()), 'max': float(norms.max()),
                    'max_unit_error': float(np.abs(norms - 1).max()), 'unit_tolerance': 0.001,
                    'zero_vectors': int(np.sum(norms == 0))}


def search(codec, v, nprobe=0, exact=False, refine=0):
    knn = {'field': 'embedding_v', 'query': v, 'k': 10, 'exact': exact,
           'ivf': {'nprobe': nprobe}, 'refine_candidates': refine}
    return {'query': {'knn': knn}, 'limit': 10, 'fields': ['id']}


def exact_truth(call, codec, vectors):
    truth = []
    for v in vectors:
        docs = call(f'/collections/{codec}/_search', search(codec, v, exact=True))['docs']
        ids = {d['id'] for d in docs}
        if len(ids) != 10:
            raise RuntimeError('Exact top ten underfilled')
        truth.append(ids)
    return truth


@contextmanager
def server(args, metric, meta):
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output.resolve() / 'data'
    if data.exists():
        raise RuntimeError('Use a fresh output directory; never reuse collection state')
    # Reserve both ports until immediately before starting the owned process.
    sockets = []
    for _ in range(2):
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        sockets.append(s)
    port, grpc = [s.getsockname()[1] for s in sockets]
    cmd = [str(Path(args.binary).resolve()), '--store.backend=fs', f'--store.data-dir={data}', f'--server.http.port={port}', f'--server.grpc.port={grpc}', f'--server.workers={args.workers}', '--max-ram-mb=4096']
    if args.max_inflight_batches:
        cmd.append(f'--indexing.max-inflight-batches={args.max_inflight_batches}')
    for s in sockets:
        s.close()
    session = requests.Session()
    base = f'http://127.0.0.1:{port}'

    def call(path, body=None):
        r = session.post(base + path, json=body, timeout=7200) if body is not None else session.get(base + path, timeout=120)
        r.raise_for_status()
        return checked(r.json())

    cpu = Path('/proc/cpuinfo').read_text()
    (args.output / 'cpuinfo.txt').write_text(cpu)
    result = {'label': args.label, 'source_revision': args.source_revision, 'binary_sha256': hashlib.file_digest(Path(args.binary).open('rb'), 'sha256').hexdigest(), 'harness_sha256': hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest(), 'dataset': meta, 'metric': metric, 'index_count': meta['index_count'], 'query_count': meta['query_count'], 'k': 10, 'workers': args.workers, 'max_inflight_batches': args.max_inflight_batches, 'runner': {k: os.getenv(k) for k in ['RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH', 'GITHUB_RUN_ID']}, 'host': platform.uname()._asdict(), 'cpu_model': next((x.split(':', 1)[1].strip() for x in cpu.splitlines() if x.startswith('model name')), platform.machine()), 'simd': {x: x in cpu.lower() for x in ['avx2', 'avx512f', 'neon', 'asimd']}, 'command': cmd, 'rows': []}
    log = (args.output / 'server.log').open('w')
    shim_source = args.output.resolve() / 'loopback-bind.c'
    shim = args.output.resolve() / 'loopback-bind.so'
    shim_source.write_text(LOOPBACK_BIND)
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(shim_source), '-ldl', '-o', str(shim)], check=True)
    result['loopback_bind_sha256'] = hashlib.file_digest(shim.open('rb'), 'sha256').hexdigest()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': str(args.workers), 'OPENBLAS_NUM_THREADS': '1', 'LD_PRELOAD': str(shim)})
    stop_guard = threading.Event()
    result['free_disk_min'] = shutil.disk_usage(args.output).free
    result['disk_floor_bytes'] = 60 * 10**9
    def guard_disk():
        while not stop_guard.wait(1):
            free = shutil.disk_usage(args.output).free
            result['free_disk_min'] = min(result['free_disk_min'], free)
            if free < result['disk_floor_bytes']:
                result['disk_guard_failed'] = True
                proc.terminate()
                return
    monitor = threading.Thread(target=guard_disk, daemon=True)
    monitor.start()
    try:
        deadline = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                raise RuntimeError('Server exited; see server.log')
            try:
                call('/collections')
                break
            except requests.RequestException:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        yield call, session, base, data, result
    finally:
        stop_guard.set()
        monitor.join(timeout=2)
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        session.close()
        listeners_closed = {}
        for listener in [port, grpc]:
            with socket.socket() as probe:
                probe.settimeout(1)
                listeners_closed[str(listener)] = probe.connect_ex(('127.0.0.1', listener)) != 0
        (args.output / 'shutdown.json').write_text(json.dumps({'exit_code': proc.returncode, 'listeners_closed': listeners_closed,
            'free_disk_min': result['free_disk_min'], 'disk_guard_failed': result.get('disk_guard_failed', False)}, indent=2))
        if result.get('disk_guard_failed'):
            raise RuntimeError('Aborted owned server: free disk below 60 GB')

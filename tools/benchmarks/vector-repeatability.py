#!/usr/bin/env python3
"""Repeat identical HTTP ingests, retain small manifests, delete owned collections."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import time

import requests


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def files(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        item = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        if p.stat().st_size >= 48:
            with p.open('rb') as f:
                f.seek(-48, 2)
                footer = struct.unpack('<4q4i', f.read(48))
                if footer[-1] == 0x4C495853:
                    f.seek(0)
                    item['faiss_header_sha256'] = hashlib.sha256(f.read(footer[0])).hexdigest()
                    item['footer'] = footer
        result[str(p.relative_to(root))] = item
    return result


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--serial', action='store_true')
    p.add_argument('--premerge', action='store_true')
    p.add_argument('--frozen-input', action='store_true', help='Reuse the first merged segment, before any ANN build')
    p.add_argument('--codec', choices=['default', 'pq'], default='default')
    p.add_argument('--queries', type=int, default=1000)
    p.add_argument('--assert-repeatable', action='store_true')
    p.add_argument('--centroid-extractor', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('compare', a.source / 'tools/benchmarks/vector-compare.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    queries, stream, metric, meta = compare.dataset(a.cache)
    queries = queries[:a.queries].tolist()
    revision = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'source_revision': revision, 'binary_sha256': sha(a.binary), 'dataset': meta,
              'metric': metric, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'harness_sha256': sha(Path(__file__)), 'builds': []}
    shim_c = a.output.resolve() / 'loopback.c'
    shim_so = a.output.resolve() / 'loopback.so'
    shim_c.write_text(compare.LOOPBACK_BIND)
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(shim_c), '-ldl', '-o', str(shim_so)], check=True)
    truth = None
    snapshot = a.output / 'frozen-data'
    try:
        for repeat in range(a.repeats):
            free = shutil.disk_usage(a.output).free
            if free < 60 * 10**9:
                raise RuntimeError('Free disk below 60 GB')
            out = a.output / str(repeat)
            out.mkdir()
            data = out.resolve() / 'data'
            if a.frozen_input and repeat:
                shutil.copytree(snapshot, data)
            sockets = [socket.socket(), socket.socket()]
            for s in sockets:
                s.bind(('127.0.0.1', 0))
            port, grpc = [s.getsockname()[1] for s in sockets]
            cmd = [str(a.binary.resolve()), '--store.backend=fs', f'--store.data-dir={data}',
                   f'--server.http.port={port}', f'--server.grpc.port={grpc}',
                   f'--server.workers={a.workers}', '--max-ram-mb=4096']
            if a.serial:
                cmd.append('--indexing.max-inflight-batches=1')
            log = (out / 'server.log').open('w')
            for s in sockets:
                s.close()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS': str(a.workers),
                                         'OPENBLAS_NUM_THREADS': '1', 'LD_PRELOAD': str(shim_so)})
            session = requests.Session()
            url = f'http://127.0.0.1:{port}'
            record = {'repeat': repeat, 'command': cmd, 'free_disk_start': free}
            result['builds'].append(record)
            def call(path, body=None):
                r = session.get(url + path, timeout=120) if body is None else session.post(url + path, json=body, timeout=1200)
                r.raise_for_status()
                return compare.checked(r.json())
            def search(v, exact=False):
                return {'query': {'knn': {'field': 'embedding_v', 'query': v, 'k': 10, 'exact': exact}},
                        'limit': 10, 'fields': ['id'], 'get_scores': True}
            try:
                deadline = time.monotonic() + 120
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError('Server exited; see server.log')
                    try:
                        call('/collections')
                        break
                    except requests.RequestException:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.2)
                settings = {} if a.codec == 'default' else {'ann': {'codec': 'pq'}}
                schema = {'fields': {'embedding_v': {'type': 'vector', 'dims': meta['dims'], 'metric': metric, **settings}}}
                record['schema'] = schema
                if not (a.frozen_input and repeat):
                    call('/collections/repeat/_schema', schema)
                    started = time.monotonic()
                    with stream.open('rb') as f:
                        r = session.post(url + '/collections/repeat/_update', data=f,
                                         headers={'Content-Type': 'application/x-ndjson'}, timeout=1200)
                    r.raise_for_status()
                    for line in r.text.splitlines():
                        if line:
                            compare.checked(json.loads(line))
                    record['ingest_s'] = time.monotonic() - started
                    record['prebuild_stats'] = call('/collections/repeat/_stats?files=true&segments=true')
                    if a.premerge or a.frozen_input:
                        call('/collections/repeat/_update', {'commit': {'max_segments': 1, 'wait_for_merges': True}})
                        record['premerged_stats'] = call('/collections/repeat/_stats?files=true&segments=true')
                if a.frozen_input and repeat == 0:
                    shutil.copytree(data, snapshot)
                record['ann_input_files'] = files(data / 't/t0/repeat')
                started = time.monotonic()
                call('/collections/repeat/_update', {'commit': {'build_aux_indexes': ['*'], 'max_segments': 1, 'wait_for_merges': True}})
                record['build_s'] = time.monotonic() - started
                record['stats'] = call('/collections/repeat/_stats?files=true&segments=true')
                index = record['stats']['collections'][0]['shards'][0]['index']
                assert len(index['segments']) == 1 and int(index['totals']['live_docs']) == meta['index_count']
                assert index['segments'][0]['overlays'], 'Refusing exact fallback'
                record['files'] = files(data / 't/t0/repeat')
                if a.centroid_extractor:
                    record['centroids'] = {}
                    for name, item in record['files'].items():
                        if 'footer' not in item:
                            continue
                        raw = subprocess.check_output([str(a.centroid_extractor.resolve()), str(data / 't/t0/repeat' / name)])
                        count, = struct.unpack('<Q', raw[:8])
                        record['centroids'][name] = {
                            'coarse_bytes': count, 'coarse_sha256': hashlib.sha256(raw[8:8+count]).hexdigest(),
                            'pq_bytes': len(raw)-8-count, 'pq_sha256': hashlib.sha256(raw[8+count:]).hexdigest()}
                record['resolved_schema'] = call('/collections/repeat/_schema?view=resolved')
                # A constant-score match-all query orders ties by internal document ID.
                r = session.post(url + '/collections/repeat/_search',
                                 json={'query': '*:*', 'limit': meta['index_count'], 'fields': ['id']}, timeout=120)
                r.raise_for_status()
                order_docs = [d for line in r.text.splitlines() if line
                              for d in compare.checked(json.loads(line)).get('docs', [])]
                order = [d['id'] for d in order_docs]
                assert len(order) == meta['index_count'] and len(set(order)) == len(order)
                record['document_order'] = {'sha256': digest(order), 'first64': order[:64],
                                            'adjacent_inversions': sum(int(x) > int(y) for x, y in zip(order, order[1:]))}
                if truth is None:
                    print(f'{repeat}: exact truth', flush=True)
                    truth = [set(d['id'] for d in call('/collections/repeat/_search', search(v, True))['docs']) for v in queries]
                    assert all(len(x) == 10 for x in truth)
                    write(a.output / 'truth.json', [sorted(x) for x in truth])
                record['query_passes'] = []
                for pass_no in range(2):
                    responses = [call('/collections/repeat/_search', search(v))['docs'] for v in queries]
                    assert all(len(x) == 10 for x in responses)
                    recall = sum(len({d['id'] for d in docs} & expected) for docs, expected in zip(responses, truth)) / (10 * len(queries))
                    record['query_passes'].append({'recall_at_10': recall, 'docs_scores_sha256': digest(responses)})
                record['delete_response'] = call('/collections/_delete', {'name': 'repeat'})
                print(f'{repeat}: recall={record["query_passes"]} order={record["document_order"]["sha256"]}', flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log.close()
                session.close()
                record['exit_code'] = proc.returncode
                shutil.rmtree(data, ignore_errors=True)
                record['data_removed'] = not data.exists()
                record['free_disk_end'] = shutil.disk_usage(a.output).free
                write(a.output / 'results.json', result)
    finally:
        if snapshot.exists():
            shutil.rmtree(snapshot)
    if a.assert_repeatable:
        signatures = [(r['document_order']['sha256'],
                       sorted(v['sha256'] for v in r['files'].values() if 'footer' in v),
                       r['query_passes']) for r in result['builds']]
        assert all(x == signatures[0] for x in signatures), 'Index or query results changed across identical builds'


if __name__ == '__main__':
    main()

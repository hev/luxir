#!/usr/bin/env python3
"""Single-binary production-policy ANN build/query grid. See vector-grid.md."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import struct
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import requests

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vector_compare', HERE / 'vector-compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
PRESETS = json.loads((HERE / 'vector-grid-datasets.json').read_text())
NPROBES = [0, 8, 16, 32, 64, 128, 256]
REFINES = [0, 10, 50, 100, 200, 500, 1000]
SOURCE = 'a7ba8db063060f25d9c34f31c9d98899fc26ed11'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def pin_valid(preset):
    if not re.fullmatch('[0-9a-f]{40}', preset['revision']):
        raise ValueError('Dataset revision must be an immutable 40-digit SHA')
    if not preset['files'] or preset['dims'] < 1 or preset['index_count'] < 10:
        raise ValueError('Invalid dataset shape/files')
    paths = []
    for f in preset['files']:
        p = Path(f['path'])
        if p.is_absolute() or '..' in p.parts or not re.fullmatch('[0-9a-f]{64}', f['sha256']):
            raise ValueError('Invalid dataset file pin')
        paths.append(f['path'])
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate dataset file')


def disk_guard(path):
    free = shutil.disk_usage(path).free
    if free < 60 * 10**9:
        raise RuntimeError(f'Aborting cleanly: free disk {free} < 60 GB')
    return free


def automatic_m(dims, metric, codec='pq'):
    # VectorIndexBuilder::choosePqM at SOURCE; explicit values are never rounded.
    cap = max(32, dims // 8) if metric in ('ip', 'cosine') else 32
    if codec == 'pq4':
        cap *= 2
    for minimum_width in (4, 1):
        for m in range(min(cap, dims), 0, -1):
            if dims % m == 0 and dims // m >= minimum_width:
                return m
    raise ValueError('Invalid dims')


def build_grid(dims, metric, smoke=False):
    builds = [{'name': 'default', 'settings': {}}]
    skipped = []
    if smoke:
        return builds + [{'name': 'pq4-auto', 'settings': {'ann': {'codec': 'pq4'}}}], skipped
    for codec in ('pq', 'pq4'):
        if codec == 'pq4' and metric == 'l2':
            skipped.append({'name': 'pq4', 'reason': 'Luxir pq4 rejects L2'})
            continue
        auto = automatic_m(dims, metric, codec)
        for numerator, denominator in ((1, 2), (1, 1), (2, 1)):
            name = f'{codec}-{numerator}over{denominator}'
            if auto * numerator % denominator:
                skipped.append({'name': name, 'reason': 'Nonintegral m; no rounding'})
                continue
            m = auto * numerator // denominator
            if dims % m:
                skipped.append({'name': name, 'm': m, 'reason': 'm does not divide dims'})
                continue
            if metric == 'l2' and m > 32:
                skipped.append({'name': name, 'm': m, 'reason': 'L2 retains m <= 32 cap'})
                continue
            builds.append({'name': name, 'automatic_m': auto, 'settings': {
                'ann': {'codec': codec, 'pq': {'m': m, 'bits': 4 if codec == 'pq4' else 8}}}})
    builds += [{'name': f'rabitq-{b}', 'settings': {'ann': {'codec': 'rabitq', 'rabitq': {'bits': b}}}}
               for b in (1, 2, 4)]
    return builds, skipped


def query_grid(smoke=False):
    return list(itertools.product([0] if smoke else NPROBES, [0] if smoke else REFINES))


def select(rows, target=0.95):
    eligible = [r for r in rows if r['recall_at_10'] >= target]
    cheapest = min(eligible, key=lambda r: (r['p50_ms'], r['index_bytes'], r['build']), default=None)
    front = [r for r in rows if not any(
        q['p50_ms'] <= r['p50_ms'] and q['recall_at_10'] >= r['recall_at_10'] and
        (q['p50_ms'] < r['p50_ms'] or q['recall_at_10'] > r['recall_at_10']) for q in rows)]
    return cheapest, sorted(front, key=lambda r: (r['p50_ms'], -r['recall_at_10']))


def fvecs(path, dims):
    a = np.memmap(path, dtype='<i4', mode='r')
    if a.size % (dims + 1):
        raise RuntimeError('Truncated fvecs')
    a = a.reshape(-1, dims + 1)
    if not np.all(a[:, 0] == dims):
        raise RuntimeError('Wrong fvecs dimensions')
    return a[:, 1:].view('<f4')


def prepare(cache, name):
    if name in compare.PRESETS:
        queries, stream, metric, meta = compare.dataset(cache, name)
        # Load base vectors for the independent numpy spot check, bounded memory.
        a = np.lib.format.open_memmap(cache / name / 'grid-vectors.npy', mode='w+', dtype='float32',
                                    shape=(meta['index_count'] + len(queries), meta['dims']))
        for i, line in enumerate(stream.open()):
            if i == meta['index_count']:
                break
            a[i] = json.loads(line)['embedding_v']
        a[meta['index_count']:] = queries
        a.flush()
        return a[:meta['index_count']], queries, stream, metric, meta
    p = PRESETS[name]
    pin_valid(p)
    cache = cache / name
    cache.mkdir(parents=True, exist_ok=True)
    disk_guard(cache)
    manifest = cache / 'source.json'
    if manifest.exists() and json.loads(manifest.read_text()) != p:
        raise RuntimeError('Dataset cache source differs from pinned preset')
    write(manifest, p)
    paths = {}
    for f in p['files']:
        path = cache / f['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        disk_guard(cache)
        compare.download(f'https://huggingface.co/datasets/{p["dataset"]}/resolve/{p["revision"]}/{f["path"]}', path, f['sha256'], guard=lambda: disk_guard(cache))
        paths[f['path']] = path
    count, dims = p['index_count'] + p['query_count'], p['dims']
    # Always recreate converted vectors from verified source files. Never trust a stale conversion.
    a = np.lib.format.open_memmap(cache / 'vectors.npy', mode='w+', dtype='float32', shape=(count, dims))
    offset = 0
    if p['format'] == 'parquet':
        for path in paths.values():
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=[p['vector_column']]):
                take = min(batch.num_rows, count - offset)
                col = batch.column(0).slice(0, take)
                if not np.all(col.value_lengths().to_numpy() == dims):
                    raise RuntimeError('Wrong vector dimensions')
                a[offset:offset + take] = col.flatten().to_numpy().reshape(-1, dims)
                offset += take
                if offset == count:
                    break
            if offset == count:
                break
    else:
        if p['format'] == 'texmex':
            with tarfile.open(next(iter(paths.values()))) as archive:
                for key in ('base_member', 'query_member'):
                    dest = cache / Path(p[key]).name
                    with archive.extractfile(p[key]) as src, dest.open('wb') as out:
                        shutil.copyfileobj(src, out)
            base = fvecs(cache / Path(p['base_member']).name, dims)
            queries = fvecs(cache / Path(p['query_member']).name, dims)
        else:
            base = np.memmap(paths[p['base_file']], dtype='<f4', mode='r').reshape(-1, dims)
            queries = np.memmap(paths[p['query_file']], dtype='<f4', mode='r').reshape(-1, dims)
        if len(base) != p['index_count'] or len(queries) < p['query_count']:
            raise RuntimeError('Unexpected base/query row counts')
        a[:len(base)] = base
        a[len(base):] = queries[:p['query_count']]
        offset = count
    if offset != count:
        raise RuntimeError(f'Unexpected rows: {offset} != {count}')
    a.flush()
    norms = []
    for start in range(0, count, 4096):
        _, stats = compare.validate_vectors(a[start:start + 4096], dims)
        norms.append(stats)
    stats = {'min': min(x['min'] for x in norms), 'max': max(x['max'] for x in norms),
             'max_unit_error': max(x['max_unit_error'] for x in norms), 'unit_tolerance': 0.001}
    metric = p['metric']
    if metric == 'auto':
        metric = 'ip' if stats['max_unit_error'] <= 0.001 else 'cosine'
    meta = {**p, 'preset': name, 'norms': stats, 'observed_metric': metric,
            'observed_rows': count, 'observed_dims': dims}
    stream = cache / 'train.ndjson'
    # Recreate from pinned input on every invocation; one conversion for all builds.
    with stream.with_suffix('.partial').open('w') as f:
        for i in range(p['index_count']):
            if i % 10000 == 0:
                disk_guard(cache)
            f.write(json.dumps({'id': str(i), 'embedding_v': a[i].tolist()}, separators=(',', ':'), allow_nan=False) + '\n')
        f.write('{"_end_":{"commit":{}}}\n')
    stream.with_suffix('.partial').replace(stream)
    with stream.open('rb') as f:
        meta['ndjson_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    write(cache / 'observed.json', meta)
    return a[:p['index_count']], a[p['index_count']:].copy(), stream, metric, meta


def spot_check(base, queries, truth, metric):
    reports = []
    # Deterministic, spread across the held-out query set. Chunked float64 scores.
    for qi in np.linspace(0, len(queries) - 1, 20, dtype=int):
        q = queries[qi].astype('float64')
        scores = np.empty(len(base), dtype='float64')
        for start in range(0, len(base), 8192):
            x = base[start:start + 8192].astype('float64')
            if metric == 'l2':
                score = -np.sum((x - q) ** 2, axis=1)
            else:
                score = x @ q
                if metric == 'cosine':
                    score /= np.linalg.norm(x, axis=1) * np.linalg.norm(q)
            scores[start:start + len(x)] = score
        top = np.argpartition(scores, -10)[-10:]
        actual = {str(i) for i in top}
        expected = truth[qi]
        # SIFT can have exact ties. Accept only score-equivalent boundary IDs.
        cutoff = scores[top].min()
        tolerance = max(1e-6, abs(cutoff) * 1e-6)
        match = actual == expected
        tied = all(scores[int(i)] >= cutoff - tolerance for i in expected)
        reports.append({'query': int(qi), 'numpy_ids': sorted(actual), 'luxir_ids': sorted(expected),
                        'same_ids': match, 'score_equivalent': tied, 'cutoff': float(cutoff), 'tolerance': tolerance})
        if not tied:
            raise RuntimeError(f'Luxir exact disagrees with numpy at query {qi}')
    return reports


def observed_ann(directory, dims, evidence=None):
    # AuxIndexInfo.opaque_meta contains the fixed 8-int VectorAuxMeta on-disk
    # contract. Preserve the commit bytes and offsets, rather than claim zeros
    # in the resolved schema are materialized build parameters.
    candidates = []
    for path in directory.rglob('commit_*'):
        raw = path.read_bytes()
        if evidence is not None:
            dest = evidence / path.relative_to(directory)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        needle = struct.pack('<i', dims)
        offset = raw.find(needle)
        while offset >= 0:
            if offset + 32 <= len(raw):
                d, metric, norm, engine, nlist, m, bits, rb = struct.unpack_from('<8i', raw, offset)
                if metric in (0, 1, 2, 3) and norm in (0, 1) and engine in (1, 2, 3) and 1 <= nlist <= 4096 and 0 <= m <= dims and bits in (0, 4, 8) and rb in (0, 1, 2, 4):
                    candidates.append({'codec': {1: 'pq', 2: 'rabitq', 3: 'pq4'}[engine], 'dims': d,
                        'metric_enum': metric, 'nlist': nlist, 'pq': {'m': m, 'bits': bits},
                        'rabitq': {'bits': rb}, 'commit_file': str(path.relative_to(directory)),
                        'offset': offset, 'commit_sha256': hashlib.sha256(raw).hexdigest(),
                        'opaque_meta_hex': raw[offset:offset + 32].hex()})
            offset = raw.find(needle, offset + 1)
    values = {(x['codec'], x['nlist'], x['pq']['m'], x['pq']['bits'], x['rabitq']['bits']) for x in candidates}
    if len(values) != 1:
        raise RuntimeError(f'Cannot unambiguously read built ANN metadata: {candidates}')
    return candidates[-1]


def measure(session, url, payloads, truth):
    times, recalls = [], []
    for payload, expected in zip(payloads, truth):
        start = time.perf_counter_ns()
        r = session.post(url, data=payload, headers={'Content-Type': 'application/json'}, timeout=120)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        r.raise_for_status()
        ids = {d['id'] for d in compare.checked(r.json())['docs']}
        if len(ids) != 10:
            raise RuntimeError('ANN top ten underfilled')
        times.append(elapsed)
        recalls.append(len(ids & expected) / 10)
    return {'p50_ms': float(np.percentile(times, 50)), 'p99_ms': float(np.percentile(times, 99)),
            'recall_at_10': statistics.mean(recalls), 'latencies_ms': times}


def throughput(url, payloads, truth):
    def worker(items):
        with requests.Session() as session:
            return measure(session, url, [x[0] for x in items], [x[1] for x in items])
    pairs = list(zip(payloads, truth))
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, [pairs[i::8] for i in range(8)]))
    elapsed = time.perf_counter() - start
    return {'workers': 8, 'queries': len(payloads), 'elapsed_s': elapsed, 'qps': len(payloads) / elapsed,
            'recall_at_10': statistics.mean(x['recall_at_10'] for x in results)}


def summarize(result, output):
    rows = result['rows']
    default = next((r for r in rows if r['build'] == 'default' and r['nprobe'] == r['refine_candidates'] == 0), None)
    cheapest, front = select(rows)
    result['selection'] = {'target_recall_at_10': 0.95, 'current_default': default,
                           'cheapest_by_p50': cheapest, 'pareto_front': front,
                           'pareto_axes': ['min p50_ms', 'max recall_at_10']}
    write(output / 'results.json', result)
    lines = [f'## {result["dataset"]["preset"]}: stage 1', '',
             f'Indexed {result["dataset"]["index_count"]}; queries {result["dataset"]["query_count"]}.', '',
             '| Selection | Build | nprobe | refine | Recall@10 | p50 ms | p99 ms | bytes/vector |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, selected in [('Current default', [default]), ('Cheapest >= 0.95', [cheapest]), ('Pareto', front)]:
        for row in selected:
            if row is None:
                lines.append(f'| {label} | no qualifying point | | | | | | |')
            else:
                lines.append(f'| {label} | {row["build"]} | {row["nprobe"]} | {row["refine_candidates"]} | {row["recall_at_10"]:.4f} | {row["p50_ms"]:.3f} | {row["p99_ms"]:.3f} | {row["bytes_per_vector"]:.1f} |')
    lines += ['', 'Adaptive settings are zero. Median of three serial passes, after one warm-up.',
              'Pareto axes: lower p50 and higher recall. Index bytes include stored vectors and documents.',
              'See results.json for effective schema, observed overlay parameters, load, build costs, passes and skips.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    disk_guard(args.cache)
    base_vectors, queries, stream, metric, meta = prepare(args.cache, args.dataset)
    builds, skips = build_grid(meta['dims'], metric, args.smoke)
    result = {'dataset': meta, 'metric': metric, 'source_revision': SOURCE,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'binary_sha256': hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
              'rows': [], 'builds': [], 'skipped': skips, 'completed': False}
    truth = None
    vectors = queries.tolist()
    for build in builds:
        disk_guard(args.cache)
        name = build['name']
        out = args.output / name
        config = SimpleNamespace(binary=args.binary, output=out, workers=args.workers,
                                 label=name, source_revision=SOURCE)
        record = {**build, 'load_start': os.getloadavg(), 'free_disk_start': disk_guard(args.cache)}
        result['builds'].append(record)
        print(f'{name}: starting', flush=True)
        try:
            with compare.server(config, metric, meta) as (call, session, url, data, server_info):
                result.setdefault('server', server_info)
                collection = 'grid'
                directory = data / 't' / 't0' / collection
                created = False
                try:
                    schema = {'fields': {'embedding_v': {'type': 'vector', 'dims': meta['dims'], 'metric': metric, **build['settings']}}}
                    record['authored_schema'] = schema
                    call(f'/collections/{collection}/_schema', schema)
                    created = True
                    start = time.perf_counter()
                    with stream.open('rb') as f:
                        r = session.post(url + f'/collections/{collection}/_update', data=f,
                                         headers={'Content-Type': 'application/x-ndjson'}, timeout=7200)
                    r.raise_for_status()
                    for line in r.text.splitlines():
                        if line:
                            compare.checked(json.loads(line))
                    visible = call(f'/collections/{collection}/_search', {'query': '*:*', 'limit': 0, 'count_limit': -1})
                    if visible.get('found') != meta['index_count']:
                        raise RuntimeError(f'Wrong visible document count: {visible}')
                    record['ingest_s'] = time.perf_counter() - start
                    start = time.perf_counter()
                    call(f'/collections/{collection}/_update', {'commit': {'build_aux_indexes': ['*'], 'max_segments': 1, 'wait_for_merges': True}})
                    record['ann_build_s'] = time.perf_counter() - start
                    stats = call(f'/collections/{collection}/_stats?files=true&segments=true')
                    write(out / 'stats.json', stats)
                    index = stats['collections'][0]['shards'][0]['index']
                    if len(index['segments']) != 1 or index['totals']['live_docs'] != meta['index_count']:
                        raise RuntimeError('Expected one complete segment')
                    overlays = index['segments'][0].get('overlays', [])
                    if not any(x.get('field') == 'embedding_v' and x.get('files') for x in overlays):
                        raise RuntimeError('Missing ANN overlay; refusing exact fallback')
                    record['effective_schema'] = call(f'/collections/{collection}/_schema?view=resolved')
                    record['observed_ann'] = observed_ann(directory, meta['dims'], out / 'commit-metadata')
                    sizes = {str(p.relative_to(directory)): p.stat().st_size for p in directory.rglob('*') if p.is_file()}
                    write(out / 'files.json', sizes)
                    record['committed_bytes'] = int(index['totals']['committed_bytes'])
                    record['index_bytes'] = sum(sizes.values())
                    record['overlay_bytes'] = sum(x['bytes'] for x in overlays if x.get('field') == 'embedding_v')
                    record['bytes_per_vector'] = record['index_bytes'] / meta['index_count']
                    if truth is None:
                        print(f'{name}: exact ground truth', flush=True)
                        truth = compare.exact_truth(call, collection, vectors)
                        write(args.output / 'truth.json', [sorted(x) for x in truth])
                        write(args.output / 'numpy-spot-check.json', spot_check(base_vectors, queries, truth, metric))
                    search_url = url + f'/collections/{collection}/_search'
                    for nprobe, refine in query_grid(args.smoke):
                        disk_guard(args.cache)
                        payloads = [json.dumps(compare.search(collection, v, nprobe=nprobe, refine=refine), separators=(',', ':')).encode() for v in vectors]
                        measure(session, search_url, payloads, truth)  # warm-up
                        passes = [measure(session, search_url, payloads, truth) for _ in range(3)]
                        row = {'build': name, 'nprobe': nprobe, 'refine_candidates': refine,
                               **{k: record[k] for k in ('ingest_s', 'ann_build_s', 'index_bytes', 'bytes_per_vector', 'overlay_bytes', 'committed_bytes')},
                               **{k: statistics.median(p[k] for p in passes) for k in ('p50_ms', 'p99_ms', 'recall_at_10')},
                               'passes': passes}
                        result['rows'].append(row)
                        if nprobe == refine == 0:
                            record['default_query_throughput'] = throughput(search_url, payloads, truth)
                        summarize(result, args.output)
                        print(f'{name}: nprobe={nprobe} refine={refine} recall={row["recall_at_10"]:.4f} p50={row["p50_ms"]:.3f}', flush=True)
                finally:
                    # Use the authored collection administration API. The outer
                    # cleanup waits for server exit before removing owned data.
                    if created:
                        record['delete_response'] = call('/collections/_delete', {'name': collection})
        except requests.HTTPError as error:
            # Record only schema/build rejections as skips. Query/measurement failures
            # invalidate the job rather than silently shrinking a completed grid.
            if error.response.status_code == 400 and 'ann_build_s' not in record and name != 'default':
                record['rejected'] = error.response.text
                result['skipped'].append({'name': name, 'reason': error.response.text})
            else:
                raise
        finally:
            record['load_end'] = os.getloadavg()
            record['free_disk_end'] = shutil.disk_usage(args.cache).free
            if (out / 'data').exists():
                shutil.rmtree(out / 'data')
            summarize(result, args.output)
    result['completed'] = True
    summarize(result, args.output)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write((args.output / 'summary.md').read_text())


def main():
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--dataset', required=True, choices=[*PRESETS, *compare.PRESETS])
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--stage', choices=['1'], default='1')
    p.add_argument('--smoke', action='store_true', help='100k control: default/pq4-auto, adaptive queries only')
    args = p.parse_args()
    if args.smoke and args.dataset != compare.DEFAULT_DATASET:
        p.error('Smoke uses the existing dbpedia-openai3-1536-100k preset')
    run(args)


if __name__ == '__main__':
    main()

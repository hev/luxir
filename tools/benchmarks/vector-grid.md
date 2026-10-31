# Vector defaults grid

A single Luxir binary measures the shipped ANN build and query defaults and
finds the fastest measured p50 point with median recall@10 at least 0.95.
This is a stage-1 experiment, not a new engine default recommendation.

## Run

Use the fork-only `Vector defaults grid dispatch (Oven)` workflow on
`hev/luxir:verify/defaults-grid`. Select `dataset=all`, `stage=1`.
The new workflow path is not registered on the default branch. To preserve
fork main, the already registered `vector-compare.yml` path on this branch is
a dispatch-only adapter calling the local reusable `vector-grid.yml`. Both
define dataset/stage dispatch inputs; only the selected branch is changed.
Dispatch with `gh workflow run vector-compare.yml -R hev/luxir --ref verify/defaults-grid -f dataset=all -f stage=1`.

The dataset matrix is ordered SIFT, GIST, Cohere 768, Cohere v3 1024,
DBpedia 1536, DBpedia 3072, with max-parallel 1 and fail-fast disabled.
Jobs run only on the existing `[self-hosted, linux, x64, oven-luxir]` runner.
A dataset job can take up to 24 hours (SIFT: 12 hours).

For a manual run with an existing binary:

```sh
python3 -m venv /tmp/vector-grid-venv
/tmp/vector-grid-venv/bin/pip install -r tools/benchmarks/requirements.txt
OPENBLAS_NUM_THREADS=1 /tmp/vector-grid-venv/bin/python tools/benchmarks/vector-grid.py \
  --binary /path/to/luxir --dataset sift-128-l2 \
  --cache /path/to/dedicated/datasets --output /path/to/new/results --stage 1
```

`run-vector-grid` builds source
`a7ba8db063060f25d9c34f31c9d98899fc26ed11` using the published #187 dev image
`ghcr.io/hev/luxir-dev@sha256:e6fd4747b2a258944abcf8548e05f3cce0b36a37a956bbd0a6d1d18631b27aeb`
(tag key `f863224709dcc81fd2ae32cdde5c760d2b2d85b750c4342ee885190c2eb61181`).
It never publishes an image. The dedicated source/binary/compiler cache is reused
across jobs, with a binary SHA-256 check. Its rootless UID adapter applies only
to owned dev-container invocations. No runner or Docker service is configured.

A reduced smoke uses the existing 100k preset (99k indexed, 1k held out):

```sh
OPENBLAS_NUM_THREADS=1 python tools/benchmarks/vector-grid.py \
  --binary /path/to/luxir --dataset dbpedia-openai3-1536-100k \
  --cache /path/to/dedicated/datasets --output /path/to/new/smoke --smoke
```

This measures default and automatic PQ4 builds, adaptive query settings,
one warm-up plus three measured passes, and concurrent throughput on both.

## Sources and splits

All file names, immutable source revisions, SHA-256 pins and literal card/API
license values are in `vector-grid-datasets.json`. Hashes are verified even
on cache hits; a changed cache source manifest is rejected. No dataset is
included in Git or result artifacts. Licenses are recorded as declared by each
source, without inferring terms for embeddings from the model or Wikipedia.

| Preset | Source | Indexed / held-out | License field |
|---|---|---|---|
| sift-128-l2 | fzliu/sift1m, TexMex archive mirror | 1M base; first 1k query vectors | undeclared (no card) |
| gist-960-l2 | fzliu/gist1m, TexMex archive mirror | 1M base; all 1k query vectors | undeclared (no card) |
| cohere-768-cos | YoKONCy/Cohere-1M-wikipedia-768d | 1M train; separate 1k test | apache-2.0 |
| cohere-v3-1024 | CohereLabs/wikipedia-2023-11-embed-multilingual-v3, en | rows [0,1000000); queries [1000000,1001000) | undeclared |
| dbpedia-openai3-1536 | Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | rows [0,999000); queries [999000,1000000) | mit |
| dbpedia-openai3-3072 | Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M | rows [0,999000); queries [999000,1000000) | apache-2.0 |

Parquet order is sorted shard name, then row order. The Qdrant 1M sources
have exactly 1M total rows, so the final 1k are held out per the fallback rule.
Cohere v3 needs the first eleven English shards, not only en/0000.parquet.
The 768d mirror card attributes its source to the now-deprecated
Cohere/wikipedia-22-12-en-embeddings. It provides separate normalized train/test
raw float32 files; that provenance is recorded without resolving the card's
ambiguous model naming. Source cards are the authority for those declarations.

At runtime every loaded vector is checked for dimensions, finiteness and nonzero
norm for cosine/IP. L2 permits zero vectors (present in GIST); their count is recorded.
Norm bounds and actual row counts are persisted. SIFT/GIST use L2;
Cohere 768 uses cosine; other sources use IP if every norm is within 0.001
of one, otherwise cosine. Values are converted to float32 as stored by Luxir.

## Stage 1

Each dataset starts with the exact authored default field settings `{}`. Other
builds use the vocabulary from `docs/guide/schema.md` and
`docs/guide/vector-search.md` at the pinned engine revision:

- `ann.codec=pq`, `ann.pq.bits=8`, `ann.pq.m=0.5x/1x/2x` automatic PQ m.
- `ann.codec=pq4`, `ann.pq.bits=4`, `ann.pq.m=0.5x/1x/2x` automatic PQ4 m,
  for IP/cosine. PQ4 is excluded for L2.
- `ann.codec=rabitq`, `ann.rabitq.bits=1/2/4`.

`ann.nlist` is omitted on all builds to retain automatic production policy.
Automatic m mirrors `VectorIndexBuilder::choosePqM`. L2 retains its m <= 32
cap. Nonintegral or nondivisor values are skipped with reasons, never rounded;
HTTP schema/build rejections are recorded. Default build failure fails the job.
The explicit default-equivalent PQ configuration remains a separate build.

For every build, ingest once into a fresh collection and use the same
production commit policy as vector-compare: `build_aux_indexes=["*"]`,
`max_segments=1`, `wait_for_merges=true`. Require one complete segment and a
real vector overlay. Sweep the same index with k=10, all 1k held-out queries:

- `ivf.nprobe`: 0 (adaptive), 8, 16, 32, 64, 128, 256.
- `refine_candidates`: 0 (adaptive), 10 (k), 50, 100, 200, 500, 1000.

Each point gets one warm-up pass and three serial measured passes. Persist
individual latencies and the median per-pass recall, p50 and p99. Eight workers
run one concurrent pass at query settings (0,0) for each build. Client and server
use loopback; server workers default to two. This is HTTP request latency,
including JSON/request processing, not an isolated ANN kernel measurement.

Luxir `exact:true` truth is computed once per dataset, from the first/default
collection, and reused. Twenty deterministic queries spread across the held-out
set are checked against chunked float64 NumPy brute force using the stored
float32 vectors. Score-equivalent boundary ties are allowed within a recorded
1e-6 relative/absolute score tolerance; all IDs and tolerances are retained.

The resolved schema is saved verbatim. It retains automatic m/nlist as zeros;
actual built codec/m/nlist are separately decoded from the 32-byte
`VectorAuxMeta` in commit files. The commit hash, byte offset and exact bytes
are recorded, and ambiguous metadata fails the job. This avoids presenting
inferred policy values as observed materialized settings.

Record ingest/build seconds, committed file sizes, overlay bytes, total index
bytes/vector, free disk and host load at each build start/end. Index bytes
include stored full-precision vectors and other document/index data; do not
interpret them as compressed code size. Total index and overlay sizes are
reported separately. Delete each collection through `POST /collections/_delete`,
stop the owned server, verify listeners close, then remove remaining owned
collection files even after failure. Never reuse collection state.

Dedicated downloads remain cached for stage 2. A job refuses to start a build or
query point with less than 60 GB free, checks during downloads and conversion,
and uploads partial evidence on failure. It never prunes another cache.

## Results

`results.json` is checkpointed after each query point and build; `completed`
marks whether the entire grid finished. `summary.md` and the job summary name
the current default point, the lowest measured p50 among points with median
recall >= 0.95, and the two-objective Pareto front (min p50, max recall).
A missing qualifying setting is explicit. Index bytes break equal-latency
cheapest ties. Pareto membership is not a claim of statistical significance;
shared-host load and timing noise must inform any follow-up recommendation.
Artifacts contain JSON, pass latencies, schemas, metadata bytes, file listings,
server logs and build provenance, but no vector corpus or collections.

Run tests with `python -m unittest discover -s tools/benchmarks -p 'test_*.py'`.
For actionlint, declare `oven-luxir` as a custom self-hosted label in its config.

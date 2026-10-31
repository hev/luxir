"""Grid contracts, source pins, settings and selection using real data fixtures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vector_grid', ROOT / 'vector-grid.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class GridTests(unittest.TestCase):
    def test_zero_vectors_are_valid_for_l2_only(self):
        a = np.array([[0, 0], [1, 0]], dtype='float32')
        with self.assertRaisesRegex(RuntimeError, 'Zero vector'):
            g.compare.validate_vectors(a, 2)
        _, stats = g.compare.validate_vectors(a, 2, allow_zero=True)
        self.assertEqual(stats['zero_vectors'], 1)
        self.assertEqual(stats['min'], 0)

    def test_full_query_grid(self):
        points = g.query_grid()
        self.assertEqual(len(points), 49)
        self.assertEqual(len(set(points)), 49)
        self.assertIn((0, 0), points)
        self.assertIn((256, 1000), points)
        self.assertEqual(g.query_grid(True), [(0, 0)])

    def test_production_m_policy(self):
        for dims, m in [(128, 32), (768, 96), (960, 120), (1024, 128), (1536, 192), (3072, 384)]:
            self.assertEqual(g.automatic_m(dims, 'ip'), m)
            self.assertEqual(g.automatic_m(dims, 'cosine'), m)
            self.assertLessEqual(g.automatic_m(dims, 'l2'), 32)
        self.assertEqual(g.automatic_m(1536, 'ip', 'pq4'), 384)
        self.assertEqual(g.automatic_m(7, 'ip'), 1)

    def test_build_grid(self):
        builds, skips = g.build_grid(1536, 'ip')
        self.assertEqual(len(builds), 10)
        self.assertEqual(skips, [])
        self.assertEqual(builds[0], {'name': 'default', 'settings': {}})
        self.assertEqual([b['settings']['ann']['pq']['m'] for b in builds[1:4]], [96, 192, 384])
        self.assertEqual([b['settings']['ann']['pq']['m'] for b in builds[4:7]], [192, 384, 768])
        self.assertEqual([b['settings']['ann']['rabitq']['bits'] for b in builds[7:]], [1, 2, 4])
        self.assertTrue(all('nlist' not in b['settings'].get('ann', {}) for b in builds))
        for b in builds[1:7]:
            self.assertEqual(b['settings']['ann']['pq']['bits'], 4 if b['name'].startswith('pq4') else 8)

    def test_l2_cap_and_pq4_exclusion(self):
        builds, skips = g.build_grid(960, 'l2')
        self.assertEqual(len(builds), 6)
        self.assertFalse(any(b['settings'].get('ann', {}).get('codec') == 'pq4' for b in builds))
        self.assertEqual([b['settings']['ann']['pq']['m'] for b in builds[1:3]], [16, 32])
        self.assertEqual(len(skips), 2)
        self.assertTrue(any('cap' in b['reason'] for b in skips))

    def test_invalid_half_and_double_not_rounded(self):
        builds, skips = g.build_grid(13, 'ip')
        self.assertTrue(any('Nonintegral' in b['reason'] for b in skips))
        self.assertTrue(any('divide' in b['reason'] for b in skips))
        self.assertTrue(all(13 % b['settings']['ann']['pq']['m'] == 0 for b in builds[1:] if 'pq' in b['settings']['ann']))

    def test_settings_rendering(self):
        payload = g.compare.search('grid', [1.0, 0.0], nprobe=128, refine=10)
        decoded = json.loads(json.dumps(payload))
        self.assertEqual(decoded['query']['knn'], {'field': 'embedding_v', 'query': [1, 0],
            'k': 10, 'exact': False, 'ivf': {'nprobe': 128}, 'refine_candidates': 10})
        self.assertEqual(g.compare.search('grid', [1])['query']['knn']['refine_candidates'], 0)

    def test_pareto_cheapest_and_unattainable(self):
        def row(name, latency, recall, size=100):
            return {'build': name, 'p50_ms': latency, 'recall_at_10': recall, 'index_bytes': size}
        rows = [row('fast', 1, .8), row('target', 2, .95), row('dominated', 3, .9),
                row('accurate', 4, 1), row('slow', 5, 1)]
        cheapest, front = g.select(rows)
        self.assertEqual(cheapest['build'], 'target')
        self.assertEqual([x['build'] for x in front], ['fast', 'target', 'accurate'])
        self.assertIsNone(g.select(rows[:1])[0])
        self.assertEqual(g.select([]), (None, []))
        rows.append(row('smaller', 2, .96, 90))
        self.assertEqual(g.select(rows)[0]['build'], 'smaller')

    def test_six_pinned_datasets_and_splits(self):
        self.assertEqual(len(g.PRESETS), 6)
        for name, p in g.PRESETS.items():
            g.pin_valid(p)
            self.assertEqual(p['query_count'], 1000)
            self.assertEqual(p['index_count'], 999000 if name.startswith('dbpedia') else 1000000)
        self.assertEqual(len(g.PRESETS['cohere-v3-1024']['files']), 11)
        self.assertEqual(g.PRESETS['cohere-768-cos']['license'], 'apache-2.0')
        self.assertEqual(g.PRESETS['sift-128-l2']['license'], 'undeclared')

    def test_pin_rejections(self):
        p = copy.deepcopy(next(iter(g.PRESETS.values())))
        for revision in ['main', 'x' * 40, 'a' * 39]:
            bad = {**p, 'revision': revision}
            with self.assertRaises(ValueError):
                g.pin_valid(bad)
        for path, sha in [('../escape', 'a' * 64), ('/abs', 'a' * 64), ('file', 'x' * 64), ('file', 'a' * 63)]:
            bad = {**p, 'files': [{'path': path, 'sha256': sha}]}
            with self.assertRaises(ValueError):
                g.pin_valid(bad)
        bad = {**p, 'files': p['files'] * 2}
        with self.assertRaises(ValueError):
            g.pin_valid(bad)

    def test_download_pin_validated_on_cached_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source'
            path.write_bytes(b'fixture')
            sha = hashlib.sha256(b'fixture').hexdigest()
            self.assertEqual(g.compare.download('unused', path, sha), sha)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                g.compare.download('unused', path, sha)

    def test_fvecs_and_overlay_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'base.fvecs'
            path.write_bytes(struct.pack('<iffiff', 2, 1., 2., 2, 3., 4.))
            np.testing.assert_equal(g.fvecs(path, 2), [[1, 2], [3, 4]])
            path.write_bytes(struct.pack('<iff', 3, 1., 2.))
            with self.assertRaisesRegex(RuntimeError, 'dimensions'):
                g.fvecs(path, 2)
            (root / 'commit_1').write_bytes(b'protobuf\x1a\x20' + struct.pack('<8i', 1536, 1, 0, 1, 314, 192, 8, 0))
            observed = g.observed_ann(root, 1536)
            self.assertEqual((observed['codec'], observed['nlist'], observed['pq']['m']), ('pq', 314, 192))
            (root / 'commit_2').write_bytes(struct.pack('<8i', 1536, 1, 0, 1, 1000, 192, 8, 0))
            with self.assertRaisesRegex(RuntimeError, 'unambiguously'):
                g.observed_ann(root, 1536)

    def test_numpy_truth_checks_all_metrics(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(40, 8)).astype('float32')
        queries = rng.normal(size=(20, 8)).astype('float32')
        for metric in ['ip', 'cosine', 'l2']:
            truth = []
            for q in queries:
                if metric == 'l2':
                    scores = -np.sum((base - q) ** 2, axis=1)
                else:
                    scores = base @ q
                    if metric == 'cosine':
                        scores /= np.linalg.norm(base, axis=1) * np.linalg.norm(q)
                truth.append({str(i) for i in np.argsort(scores)[-10:]})
            self.assertEqual(len(g.spot_check(base, queries, truth, metric)), 20)
            truth[0] = {str(i) for i in range(10)}
            with self.assertRaisesRegex(RuntimeError, 'disagrees'):
                g.spot_check(base, queries, truth, metric)


if __name__ == '__main__':
    unittest.main()

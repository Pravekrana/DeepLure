from collections import defaultdict
import random


class PKSampler:
    """P identities, K distinct original groups. Singleton designs excluded explicitly."""
    def __init__(self, rows, p=4, k=2, seed=17):
        self.rows, self.p, self.k, self.seed = rows, p, k, seed
        groups = defaultdict(dict)
        for i, r in enumerate(rows):
            if r.get('design_id'):
                groups[r['design_id']].setdefault(r['capture_group'] or r['original_group'], i)
        self.groups = {d: list(v.values()) for d, v in groups.items() if len(v) >= k}
        if len(self.groups) < p or k < 2:
            raise ValueError(f'P-by-K needs {p} reliable identities with {k} distinct captures; have {len(self.groups)}')

    def batches(self, epoch=0):
        rng = random.Random(self.seed + epoch)
        ids = sorted(self.groups)
        rng.shuffle(ids)
        for start in range(0, len(ids) - self.p + 1, self.p):
            yield [i for d in ids[start:start+self.p] for i in rng.sample(self.groups[d], self.k)]

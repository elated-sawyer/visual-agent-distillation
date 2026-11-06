"""Frozen panel identities and metrics; no model/environment imports."""
import math

SPATIAL_METRICS = ('cogmap_score', 'cogmap_dir', 'cogmap_facing', 'cogmap_pos', 'cogmap_exploration_coverage')

def validate_panel(panel, records):
    if panel['scope'] not in ('monitor', 'final', 'engineering'):
        raise ValueError('Unknown evaluation scope')
    if len(records) != panel['episodes'] or not records:
        raise ValueError('Incomplete evaluation panel')
    ids = [(r['task_id'], r['repeat']) for r in records]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(tuple(x) for x in panel['slots']):
        raise ValueError('Evaluation task/repeat identity mismatch')
    keys = SPATIAL_METRICS if panel['environment'] == 'spatial' else ('success',)
    for r in records:
        for k in keys:
            x = r[k]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f'Invalid {k}: {x}')
            if k == 'success' and x not in (0, 1):
                raise ValueError('Nonbinary task success')
    groups = {'all': records}
    if panel['environment'] == 'spatial':
        groups.update({split: [r for r in records if r['split'] == split]
                       for split in ('train', 'eval') if any(r['split'] == split for r in records)})
    return {f"{panel['environment']}/{panel['scope']}/{panel['name']}/{g}/{k}":
            [float(r[k]) for r in rows] for g, rows in groups.items() for k in keys}

def validate_schedule(config):
    steps = [x['step'] for x in config['curve_evaluations']]
    if steps != list(range(config['evaluation_interval'], config['updates'] + 1, config['evaluation_interval'])):
        raise ValueError('Missing or duplicate evaluation step')
    monitor = None
    for item in config['curve_evaluations']:
        panels = item['panels']
        if len({p['name'] for p in panels}) != len(panels):
            raise ValueError('Duplicate panel name')
        for panel in panels:
            if panel['scope'] == 'final' and item['step'] != config['updates']:
                raise ValueError('Final test used before final update')
        signatures = [(p['name'], p['slots']) for p in panels if p['scope'] == 'monitor']
        if not config['curve_smoke']:
            if len(signatures) != 1 or (monitor is not None and signatures != monitor):
                raise ValueError('Monitoring identities changed across updates')
            monitor = signatures
        elif any(p['scope'] != 'engineering' for p in panels):
            raise ValueError('Smoke may not evaluate the final test')

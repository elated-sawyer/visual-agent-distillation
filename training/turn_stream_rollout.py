"""Read ready, unclaimed decisions from persistent environment lanes."""
import copy, hashlib, json, random
from pathlib import Path
from online_contract import read,digest
from turn_stream_contract import decision_dir
def ready_decisions(root):
    rows=[]
    for lane in range(16):
        p=root/'lanes'/f'{lane:02d}'/'ready.json'
        if not p.exists():continue
        pointer=read(p);assert pointer['lane_id']==lane
        d=decision_dir(root,lane,pointer['sequence'])
        if (d/'claim.json').exists():continue
        item=read(d/'request.json')
        assert digest(item)==pointer['request_sha256']
        rows.append((d,item))
    return rows

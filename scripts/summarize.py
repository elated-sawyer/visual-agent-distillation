"""Summarize complete evaluation records; never convert infrastructure failures to zeros."""
import argparse,json,math
from pathlib import Path

def summarize(data):
    if data.get('status')!='passed':raise ValueError('Evaluation did not complete')
    rows=data['records'];ids=[(r['task_id'],r['repeat']) for r in rows]
    if not rows or len(set(ids))!=len(rows):raise ValueError('Empty or duplicated panel')
    spatial='cogmap_score' in rows[0];key='cogmap_score' if spatial else 'success'
    result={}
    for split in sorted({r['split'] for r in rows}):
        values=[r[key] for r in rows if r['split']==split]
        if not all(math.isfinite(v) and 0<=v<=1 and (spatial or v in [0,1]) for v in values):raise ValueError('Invalid metric')
        result[split]={'metric':key,'episodes':len(values),'mean':sum(values)/len(values)}
        if not spatial:result[split]['successes']=sum(values)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args()
    print(json.dumps(summarize(json.loads(a.results.read_text())),indent=2))

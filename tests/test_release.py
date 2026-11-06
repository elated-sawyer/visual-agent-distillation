import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from mosaic_distill.records import decision_messages,export_prefixes
from summarize import summarize

class ReleaseTests(unittest.TestCase):
    def test_sft_attachment_and_frozen_evaluation_schedule(self):
        import hashlib
        from evaluation_contract import validate_schedule
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);opd=base/'opd';labels=base/'labels';(labels/'labels').mkdir(parents=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_data.py'),'--demo','--student','/unused/model',
                            '--output',str(opd),'--updates','100','--batch-size','1'],check=True,capture_output=True)
            rows=[json.loads(x) for x in (opd/'prefixes.jsonl').read_text().splitlines()];sealed={}
            for i,row in enumerate(rows):
                md=row['metadata'];lab={k:md[k] for k in ['draw_index','pool_index','prefix_sha256']}
                path=labels/f'labels/{i:04d}.json';path.write_text(json.dumps(lab))
                sealed[f'labels/{i:04d}.json']=hashlib.sha256(path.read_bytes()).hexdigest()
            (labels/'audit.json').write_text(json.dumps({'status':'passed','labels':len(rows)}))
            (labels/'labels-sha256.json').write_text(json.dumps(sealed))
            out=base/'sft'
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_sft.py'),'attach','--opd-run',str(opd),
                            '--labels',str(labels),'--output',str(out)],check=True,capture_output=True)
            actual=[json.loads(x) for x in (out/'prefixes.jsonl').read_text().splitlines()]
            for i,(a,b) in enumerate(zip(rows,actual,strict=True)):
                self.assertEqual(a['messages'],b['messages']);self.assertEqual(b['metadata']['sft_order'],i)
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_evaluation.py'),'--run',str(opd),
                            '--panels',str(ROOT/'configs/evaluations/sokoban.json'),'--inference-ip','127.0.0.1'],
                           check=True,capture_output=True)
            cfg=json.loads((opd/'config.json').read_text());validate_schedule(cfg)
            self.assertEqual(len(cfg['curve_evaluations']),20)
            self.assertEqual(cfg['curve_evaluations'][-1]['step'],100)
            (labels/'labels/0000.json').write_text('{}')
            bad=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_sft.py'),'attach','--opd-run',str(opd),
                                '--labels',str(labels),'--output',str(base/'bad')],capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse((base/'bad').exists())

    def test_demo_is_replayable_and_preserves_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'opd'
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_data.py'),'--demo','--student','/unused/model',
                            '--output',str(out),'--updates','2','--batch-size','64'],check=True,capture_output=True)
            pool=[json.loads(x) for x in (out/'pool.jsonl').read_text().splitlines()]
            draws=[json.loads(x) for x in (out/'prefixes.jsonl').read_text().splitlines()]
            self.assertEqual(len(pool),10);self.assertEqual(len(draws),128)
            self.assertEqual(len({r['metadata']['episode_sha256'] for r in pool}),4)
            for i,r in enumerate(draws):
                self.assertEqual(r['metadata']['draw_index'],i)
                self.assertEqual(r['messages'],pool[r['metadata']['pool_index']]['messages'])
                self.assertEqual(r['messages'][-1]['role'],'user')
                for m in r['messages']:
                    for part in m['content']:
                        if part['type']=='image':self.assertTrue(Path(part['image']).is_file())
            second=Path(tmp)/'again'
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_data.py'),'--demo','--student','/unused/model',
                            '--output',str(second),'--updates','2','--batch-size','64'],check=True,capture_output=True)
            self.assertEqual((out/'prefixes.jsonl').read_bytes(),(second/'prefixes.jsonl').read_bytes())

    def test_future_response_and_outcome_not_in_prefix(self):
        def msg(text,role='user'):return {'role':role,'content':[{'type':'text','text':text}]}
        ep={'system':msg('system','system'),'turns':[{'observation':msg('obs0'),'response':'past',
              'next_observation':msg('obs1')},{'observation':msg('obs1'),'response':'SECRET FUTURE'}],'success':True}
        result=decision_messages(ep,1)
        self.assertNotIn('SECRET',json.dumps(result));self.assertIn('past',json.dumps(result))
        changed=copy.deepcopy(ep);changed['success']=False;changed['turns'][1]['response']='different'
        self.assertEqual(result,decision_messages(changed,1))

    def test_metrics_keep_denominators_and_native_scores(self):
        rows=[{'task_id':'a','repeat':0,'split':'eval','success':1},
              {'task_id':'a','repeat':1,'split':'eval','success':0}]
        self.assertEqual(summarize({'status':'passed','records':rows})['eval']['mean'],.5)
        with self.assertRaises(ValueError):summarize({'status':'passed','records':rows+[rows[0]]})
        with self.assertRaises(ValueError):summarize({'status':'failed','records':rows})
        for i,r in enumerate(rows):r['cogmap_score']=.2+.3*i
        self.assertAlmostEqual(summarize({'status':'passed','records':rows})['eval']['mean'],.35)

    def test_selection_implementation_not_shipped(self):
        for name in ['selection.py','curation.py','context_selection.py','decision_selection.py']:
            self.assertFalse((ROOT/'mosaic_distill'/name).exists())

if __name__=='__main__':unittest.main()

import copy
import unittest
from evaluation_contract import validate_panel,validate_schedule,SPATIAL_METRICS

class PanelContractTests(unittest.TestCase):
    def setUp(self):
        self.panel={'scope':'monitor','name':'heldout','environment':'spatial','episodes':2,
                    'slots':[['room-a',0],['room-a',1]]}
        self.rows=[{'task_id':'room-a','repeat':i,'split':'eval',**{k:.25+.5*i for k in SPATIAL_METRICS}}
                   for i in range(2)]
    def test_continuous_native_score_preserved(self):
        x=validate_panel(self.panel,self.rows)
        self.assertEqual(x['spatial/monitor/heldout/all/cogmap_score'],[.25,.75])
    def test_duplicate_repeat_rejected(self):
        with self.assertRaises(ValueError):validate_panel(self.panel,[self.rows[0]]*2)
    def test_wrong_identity_rejected_even_same_count(self):
        rows=copy.deepcopy(self.rows);rows[1]['task_id']='room-b'
        with self.assertRaises(ValueError):validate_panel(self.panel,rows)
    def test_missing_or_infrastructure_episode_not_zero(self):
        with self.assertRaises(ValueError):validate_panel(self.panel,self.rows[:1])
    def test_nonfinite_score_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['cogmap_score']=float('nan')
        with self.assertRaises(ValueError):validate_panel(self.panel,rows)
    def test_group_denominators_remain_separate(self):
        rows=copy.deepcopy(self.rows);rows[0]['split']='train'
        x=validate_panel(self.panel,rows)
        self.assertEqual(len(x['spatial/monitor/heldout/train/cogmap_score']),1)
        self.assertEqual(len(x['spatial/monitor/heldout/eval/cogmap_score']),1)
    def schedule(self):
        return {'updates':100,'evaluation_interval':5,'curve_smoke':False,
            'curve_evaluations':[{'step':i,'panels':[copy.deepcopy(self.panel)]} for i in range(5,101,5)]}
    def test_fixed_monitor_and_final_at100(self):
        c=self.schedule();p=copy.deepcopy(self.panel);p.update(scope='final',name='two_room')
        c['curve_evaluations'][-1]['panels'].append(p);validate_schedule(c)
    def test_final_leak_before_final_update_rejected(self):
        c=self.schedule();p=copy.deepcopy(self.panel);p.update(scope='final',name='two_room')
        c['curve_evaluations'][0]['panels'].append(p)
        with self.assertRaises(ValueError):validate_schedule(c)
    def test_changing_monitor_rejected(self):
        c=self.schedule();c['curve_evaluations'][1]['panels'][0]['slots'][0][0]='other'
        with self.assertRaises(ValueError):validate_schedule(c)
    def test_smoke_cannot_score_final_test(self):
        c={'updates':2,'evaluation_interval':1,'curve_smoke':True,
           'curve_evaluations':[{'step':i,'panels':[self.panel]} for i in [1,2]]}
        with self.assertRaises(ValueError):validate_schedule(c)

if __name__=='__main__':unittest.main()

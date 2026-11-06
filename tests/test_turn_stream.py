import unittest
from turn_stream_contract import RoundLedger,validate_response,check_execution
from online_contract import digest

class ContractTests(unittest.TestCase):
    def test_exact_credit_and_no_duplicate(self):
        ledger=RoundLedger(2)
        item={'lane_id':0,'sequence':0}
        self.assertEqual(ledger.claim(item)['sample_index'],128)
        with self.assertRaises(AssertionError):ledger.claim(item)
        for i in range(1,64):ledger.claim({'lane_id':i%16,'sequence':i})
        with self.assertRaises(AssertionError):ledger.claim({'lane_id':0,'sequence':100})
        self.assertEqual(len(ledger.claims),64)
    def test_wrong_policy_and_unexecuted_response_rejected(self):
        item={'lane_id':0,'sequence':0};claim=RoundLedger(0).claim(item)
        response={**claim,'status':'passed','response':'go','response_tokens':1}
        validate_response(item,claim,response)
        with self.assertRaises(AssertionError):validate_response(item,claim,response|{'expected_version':'2'})
        executed={'status':'passed','request_sha256':digest(item),'response_sha256':digest(response),
                  'expected_version':'1','sample_index':0,'action_executed':False,'lane_id':0,'sequence':0}
        with self.assertRaises(AssertionError):check_execution(item,claim,response,executed)

import asyncio
import unittest
from mosaic_distill.limits import remaining_from_context_error, ContextBudgetExhausted
from mosaic_distill.rollout import collect_episode


class LimitsTests(unittest.TestCase):
    def test_total_context_and_input_limit(self):
        msg = "Requested token count exceeds the model's maximum context length of 32776 tokens. You requested a total of 39192 tokens: 31000 tokens from the input messages and 8192 tokens for the completion."
        self.assertEqual(remaining_from_context_error(400, {'error': {'message': msg}}, 32768, 8192), 1768)
        msg = "The input (33000 tokens) is longer than the model's context length (32776 tokens)."
        self.assertEqual(remaining_from_context_error(400, {'error': {'message': msg}}, 32768, 8192), 0)

    def test_other_errors_never_become_task_failures(self):
        for status, data in [(500, {}), (400, {'error': {'message': 'CUDA out of memory'}}), (400, 'invalid')]:
            self.assertIsNone(remaining_from_context_error(status, data, 32768, 8192))
        msg = "Requested token count exceeds the model's limit: 100 tokens from the input messages and 8192 tokens for the completion."
        self.assertIsNone(remaining_from_context_error(400, {'error': {'message': msg}}, 32768, 8192))

    def test_exhaustion_preserves_completed_turns(self):
        class Env:
            protocol = {'max_turns': 5}
            closed = False
            async def reset(self, seed): return {'role':'user','content':[{'type':'text','text':'state'}]}, {'split':'train'}
            async def system_prompt(self): return {'role':'system','content':[{'type':'text','text':'system'}]}
            async def step(self, text): return (await self.reset(0))[0], 0, False, False, {}
            async def close(self): self.closed=True
        calls=0
        async def policy(messages):
            nonlocal calls
            calls+=1
            if calls==2: raise ContextBudgetExhausted()
            return 'original text'
        env=Env()
        result=asyncio.run(collect_episode(env,policy,0,{'kind':'student'}))
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['success'])
        self.assertEqual(result['stop_reason'],'context_limit')
        self.assertEqual(len(result['turns']),1)
        self.assertTrue(env.closed)

if __name__=='__main__': unittest.main()

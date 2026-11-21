"""Convert recorded multimodal messages into native Slime samples."""
import copy, hashlib, json, random
from pathlib import Path
def native_messages(messages, store):
    from mosaic_distill.records import validate_message
    result = copy.deepcopy(messages)
    for message in result:
        validate_message(message, role=message['role'])
        for part in message['content']:
            if part['type'] == 'image_ref':
                ref = part['image']
                store.read(ref)
                part.clear()
                part.update(type='image', image=str(store._path(ref)))
    return result

def prepare_sample(state, messages, index):
    from slime.utils.processing_utils import process_vision_info
    from slime.rollout.sglang_rollout import _prepare_prompt_ids
    from slime.utils.types import Sample
    sample = Sample(index=index, group_index=index,
                    prompt=state.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                    multimodal_inputs=process_vision_info(messages, state.processor))
    sample.tokens = list(map(int, _prepare_prompt_ids(sample, state.tokenizer, state.processor)))
    return sample

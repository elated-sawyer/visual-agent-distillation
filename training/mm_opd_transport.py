"""Use SGLang's native exact-token multimodal path for teacher scoring."""
from dataclasses import replace
from slime.rollout.on_policy_distillation import reward_func as native_reward

# Qwen3.5 shared tokenizer IDs are checked against the HF config in preflight.
VISION_START=248053
IMAGE_PAD=248056
VISION_END=248054


def collapse_image_spans(tokens,grid_thw,response_length,merge_size=2):
    """Collapse only verified prefix image spans; preserve all non-image tokens."""
    counts=[int(grid.prod().item())//(merge_size*merge_size) for grid in grid_thw]
    prefix_end=len(tokens)-response_length
    if prefix_end<=0 or response_length<=0:raise ValueError('Missing prefix/response')
    result=[];index=0;image_index=0
    while index<prefix_end:
        token=tokens[index]
        if token==VISION_START:
            if image_index>=len(counts):raise ValueError('Too many image markers')
            n=counts[image_index];end=index+1+n
            if n<=0 or end>=prefix_end or tokens[index+1:end]!=[IMAGE_PAD]*n or tokens[end]!=VISION_END:
                raise ValueError('Expanded image tokens do not match the verified grid')
            result.extend([VISION_START,IMAGE_PAD,VISION_END]);index=end+1;image_index+=1
        elif token in (IMAGE_PAD,VISION_END):
            raise ValueError('Unpaired image token in prefix')
        else:result.append(token);index+=1
    if image_index!=len(counts):raise ValueError('Image/grid count mismatch')
    response=tokens[prefix_end:]
    if any(token in (VISION_START,IMAGE_PAD,VISION_END) for token in response):
        raise ValueError('Generated visual markers are not supported by this image-only scoring path')
    result.extend(response)
    return result


async def exact_reward(args,sample,**kwargs):
    grid=sample.multimodal_train_inputs['image_grid_thw']
    ids=collapse_image_spans(sample.tokens,grid,sample.response_length)
    wire_sample=replace(sample,tokens=ids)
    # Only request encoding differs. The native receiver expands placeholders
    # while SGLANG_MM_AVOID_RETOKENIZE=1 preserves all supplied text token IDs.
    return await native_reward(args,wire_sample,**kwargs)

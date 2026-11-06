"""Runtime compatibility fix for QwenVL's missing input_ids forwarding.

The container's QwenVL processor calls legacy_load_mm_data directly, bypassing
load_mm_data, which normally preserves the original list before decoding it.
Keep the container files unchanged; only teacher launcher processes install this.
"""

def install():
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    if getattr(QwenVLImageProcessor,'_mosaic_exact_ids',False):return
    original=QwenVLImageProcessor.legacy_load_mm_data
    async def preserve(self,prompt,*args,**kwargs):
        if isinstance(prompt,list) and kwargs.get('input_ids') is None:
            kwargs['input_ids']=prompt
        return await original(self,prompt,*args,**kwargs)
    QwenVLImageProcessor.legacy_load_mm_data=preserve
    QwenVLImageProcessor._mosaic_exact_ids=True

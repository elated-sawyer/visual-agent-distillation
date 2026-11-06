"""Launch unmodified SGLang with the scoped Qwen token forwarding fix."""
from qwen_exact_ids import install
install()  # Also installed when multiprocessing spawns this entrypoint.
if __name__=='__main__':
    import runpy
    # Keep this entrypoint as multiprocessing's main module (alter_sys=False).
    runpy.run_module('sglang.launch_server',run_name='__main__')

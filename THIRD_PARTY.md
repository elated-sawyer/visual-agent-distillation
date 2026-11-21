# Third-party sources

- VAGEN: https://github.com/mll-lab-nu/VAGEN, pinned revision and source hashes in `upstreams.lock.json`. MIT notice: `licenses/VAGEN-MIT.txt`. Environment adapters call its native environment, parsing and scoring interfaces.
- gym-sokoban: https://github.com/mpSchrader/gym-sokoban. Saved Sokoban image assets and CPU environment; MIT notice: `licenses/gym-sokoban-MIT.txt`.
- Slime: https://github.com/THUDM/slime. Native rollout, OPD, SFT loss, Megatron training and asynchronous worker are external dependencies. The integration hooks use these frameworks as external dependencies.
- SGLang, Megatron-LM and Transformers are runtime dependencies with their respective upstream licenses.
- AI2-THOR: https://github.com/allenai/ai2thor. Navigation simulator binaries/assets are obtained separately, with their original terms.
- SpatialGym dataset: https://huggingface.co/datasets/yw12356/spatial_gym_dataset. Data is downloaded separately at the revision in the environment protocol; no room images are bundled.
- Qwen3.5-4B, 9B and 27B: model weights are external. Their original model terms apply; revisions are listed in `configs/models.json`.

Third-party components retain their upstream licenses. A general license for the authors' code and research artifacts has not yet been assigned.

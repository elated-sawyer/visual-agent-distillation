# Third-party sources

- VAGEN: https://github.com/mll-lab-nu/VAGEN, pinned revision and source hashes in `upstreams.lock.json`. MIT notice: `licenses/VAGEN-MIT.txt`. Environment adapters call its native environment, parsing and scoring interfaces.
- gym-sokoban: https://github.com/mpSchrader/gym-sokoban. Saved Sokoban image assets and CPU environment; MIT notice: `licenses/gym-sokoban-MIT.txt`.
- Slime: https://github.com/THUDM/slime. Native rollout, OPD, SFT loss, Megatron training and asynchronous worker are external dependencies. This repository includes integration hooks, not a redistributed training framework.
- SGLang, Megatron-LM and Transformers are runtime dependencies with their respective upstream licenses.
- AI2-THOR: https://github.com/allenai/ai2thor. Navigation simulator binaries/assets are obtained separately, with their original terms.
- SpatialGym dataset: https://huggingface.co/datasets/yw12356/spatial_gym_dataset. Data is downloaded separately at the revision in the environment protocol; no room images are bundled.
- Qwen3.5-4B, 9B and 27B: model weights are external. Their original model terms apply; revisions are listed in `configs/models.json`.

No private author-service code, environment binary, framework checkout or model weights are included. Third-party names credit upstream work and do not identify the anonymous submission's authors. No general license has yet been assigned to the authors' code and research artifacts. Inclusion/reference does not relicense third-party material.

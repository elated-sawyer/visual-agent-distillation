"""Pinned VAGEN Navigation, with project-local AI2-THOR runtime storage."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

from .records import fingerprint, validate_message
from .upstreams import ROOT, verify_upstream
from .render_guard import require_render_guard, fail_render


class VagenNavigation:
    def __init__(self, protocol, split, store, *, gpu_device=0):
        if split not in {'train', 'eval'}:
            raise ValueError('Explicit training or evaluation split required')
        if protocol.get('schema') != 'navigation-protocol-v1' or protocol['history'] != 'full':
            raise ValueError('Unknown Navigation protocol')
        if type(protocol['max_turns']) is not int or protocol['max_turns'] <= 0:
            raise ValueError('Positive outer turn limit required')
        path = verify_upstream('VAGEN').resolve()
        self.project_root = path.parents[1]
        revision = json.loads((ROOT/'upstreams.lock.json').read_text())['VAGEN']['commit']
        if protocol['upstream'] != {'name':'VAGEN', 'commit':revision}:
            raise ValueError('Upstream identity drift')
        for name, module in tuple(sys.modules.items()):
            location = getattr(module, '__file__', None)
            if (name == 'vagen' or name.startswith('vagen.')) and location:
                if not Path(location).resolve().is_relative_to(path):
                    raise RuntimeError('Foreign VAGEN already imported')
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from dataclasses import asdict
        from vagen.envs.navigation.navigation_env import NavigationEnv, NavigationEnvConfig
        from vagen.envs._common.observations import convert_obs_to_content
        from vagen.envs import TurnLimit

        cfg = copy.deepcopy(protocol['env_config'])
        if 'eval_set' in cfg or 'gpu_device' in cfg:
            raise ValueError('Dataset split and runtime GPU must be explicit adapter arguments')
        if cfg.get('down_sample_ratio', 1.0) != 1.0:
            raise ValueError('Subsampling must use frozen task indices, never implicit stride')
        descriptor = protocol['datasets'][split]
        name = descriptor['name']
        if '/' in name or '\\' in name or ('train' in name) != (split == 'train'):
            raise ValueError('Dataset does not belong to requested split')
        asset = path/'vagen/envs/navigation/assets'/f'{name}.json'
        if hashlib.sha256(asset.read_bytes()).hexdigest() != descriptor['sha256']:
            raise ValueError('Dataset content drift')
        # Dataclass rejects misspelled config rather than upstream silently ignoring it.
        config = NavigationEnvConfig(**cfg, eval_set=name, gpu_device=gpu_device)
        self.raw = NavigationEnv(asdict(config))
        if len(self.raw._dataset) != descriptor['tasks']:
            raise ValueError('Dataset size drift')
        self.env = TurnLimit(self.raw, protocol['max_turns'])
        self._convert = convert_obs_to_content
        self.protocol, self.split, self.store = copy.deepcopy(protocol), split, store
        self.done = True

    def _ensure_controller(self):
        require_render_guard()
        if self.raw._controller is not None:
            return
        runtime = self.protocol['runtime']
        if importlib.metadata.version('ai2thor') != runtime['ai2thor_version']:
            raise ValueError('AI2-THOR version drift')
        import ai2thor.build
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering

        if ai2thor.build.COMMIT_ID != runtime['build_commit']:
            raise ValueError('Unity build revision drift')
        root = Path(runtime['root']).resolve()
        if not root.is_relative_to(self.project_root):
            raise ValueError('Simulator runtime must remain in this project')
        build_name = f"thor-CloudRendering-{runtime['build_commit']}"
        if not (root/'releases'/build_name/build_name).is_file():
            raise FileNotFoundError('Offline Unity build has not been deployed')

        class ProjectController(Controller):
            @property
            def base_dir(self):
                return str(root)

        cfg = self.raw.cfg
        # Same constructor options as pinned VAGEN; only storage location changes.
        self.raw._controller = ProjectController(
            agentMode='default', gridSize=0.1, visibilityDistance=10,
            renderDepthImage=False, renderInstanceSegmentation=False,
            width=cfg.resolution, height=cfg.resolution, fieldOfView=cfg.fov,
            platform=CloudRendering, gpu_device=cfg.gpu_device,
            server_timeout=300, server_start_timeout=300,
        )

    def _check_frame(self):
        frame = self.raw._controller.last_event.frame
        if frame is None or not frame.size or not frame.any():
            fail_render('Navigation returned an empty/all-black RGB observation')

    def _message(self, obs, role='user'):
        mm = obs.get('multi_modal_input') or {}
        if any(k != '<image>' and v for k,v in mm.items()):
            raise ValueError('Unsupported observation modality')
        images = iter(mm.get('<image>', [])); content = []
        for part in self._convert(obs):
            if part['type'] == 'text':
                content.append(dict(part))
            elif part['type'] == 'image':
                content.append({'type':'image_ref', 'image':self.store.put(next(images))})
            else:
                raise ValueError('Unsupported observation content')
        if next(images, None) is not None:
            raise ValueError('Unconsumed images')
        message = {'role':role, 'content':content}
        validate_message(message, role=role)
        return message

    async def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < len(self.raw._dataset):
            raise ValueError('Task index outside requested split; modulo wrapping is forbidden')
        import asyncio
        await asyncio.to_thread(self._ensure_controller)
        obs, _ = await self.env.reset(seed=seed)
        require_render_guard()
        self._check_frame()
        # Upstream does not raise on a failed Teleport; never silently score a different start.
        if not self.raw._controller.last_event.metadata['lastActionSuccess']:
            raise RuntimeError('Native start-pose Teleport failed')
        task_sha = fingerprint(self.raw._dataset[seed])
        task = {'environment':'VAGEN/Navigation', 'split':self.split, 'seed':seed,
                'dataset':self.raw.cfg.eval_set, 'task_index':seed, 'task_sha256':task_sha,
                'scene':self.raw._dataset[seed]['scene'], 'task_id':f'navigation:{task_sha}'}
        self.done = False
        return self._message(obs), task

    async def system_prompt(self):
        return self._message(await self.env.system_prompt(), role='system')

    async def step(self, response):
        require_render_guard()
        if self.done:
            raise RuntimeError('Environment must be reset')
        obs, reward, done, info = await self.env.step(response)
        require_render_guard()
        self._check_frame()
        self.done = bool(done)
        if not math.isfinite(float(reward)):
            raise ValueError('Nonfinite native reward')
        truncated = bool(info.get('truncated', False))
        return self._message(obs), float(reward), bool(done) and not truncated, truncated, info

    async def close(self):
        self.done = True
        await self.env.close()

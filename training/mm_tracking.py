"""Explicitly finish native trainer W&B streams before Ray releases workers."""
import logging
import wandb
from slime.backends.megatron_utils.actor import MegatronTrainRayActor


class TrackingMegatronTrainRayActor(MegatronTrainRayActor):
    def train(self, rollout_id, rollout_data_ref, external_data=None):
        try:
            result=super().train(rollout_id, rollout_data_ref, external_data=external_data)
        except BaseException:
            try:
                if wandb.run is not None:wandb.finish(exit_code=1)
            except Exception:
                logging.getLogger(__name__).exception('Failed to finish trainer tracking after an error')
            raise
        if rollout_id+1==self.args.num_rollout and wandb.run is not None:
            wandb.finish(exit_code=0)
        return result

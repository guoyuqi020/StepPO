from __future__ import annotations

from verl.workers.engine_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        from arft.policy_losses import register_local_policy_losses

        register_local_policy_losses()
        super().__init__(*args, **kwargs)

    def init_model(self):
        from arft.policy_losses import register_local_policy_losses

        register_local_policy_losses()
        return super().init_model()

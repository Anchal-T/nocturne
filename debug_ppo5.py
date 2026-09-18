import os, sys
os.environ['DISPLAY']=':99'
os.environ['PROCESSED_TRAIN_NO_TL']='/home/anchal/anchal/nocturne/dataset/full_train/formatted_json_v2_no_tl_train'
os.environ['NOCTURNE_LOG_DIR']='/home/anchal/anchal/nocturne/outputs'
sys.path.insert(0, '/home/anchal/anchal/nocturne')
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
initialize(config_path="cfgs", version_base=None)
cfg = compose(config_name="config", overrides=["algorithm=ppo", "++algorithm.n_rollout_threads=1", "++algorithm.n_eval_rollout_threads=1", "++subscriber.use_occlusion_features=False", "++algorithm.use_lagrangian=True", "++rew_cfg.near_miss_threshold=2.0"])
# mimic runner: set keep_inactive_agents True
cfg.subscriber.keep_inactive_agents = True
print("keep_inactive", cfg.subscriber.keep_inactive_agents)
print("max_num", cfg.max_num_vehicles, "occlusion", cfg.subscriber.use_occlusion_features)
# Also check observation dim via cfg
from nocturne.envs.wrappers import create_ppo_env
from algos.ppo.env_wrappers import DummyVecEnv
def make_fn(rank=0):
    def _f():
        return create_ppo_env(cfg, rank=rank)
    return _f

# Test single env directly
env = create_ppo_env(cfg, rank=0)
print("env created")
obs = env.reset()
print("reset len", len(obs), "shapes", [o.shape for o in obs][:3], "...", [o.shape for o in obs][-3:])
print("unique", set([o.shape for o in obs]))
# Check DummyVecEnv
vec = DummyVecEnv([make_fn()])
print("vec created")
try:
    obs2 = vec.reset()
    print("vec shape", obs2.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    # manual
    obs_list = [env.reset() for env in vec.envs]
    print("manual obs_list", len(obs_list), len(obs_list[0]))
    for i,o in enumerate(obs_list[0]):
        print(i, o.shape, o.dtype)

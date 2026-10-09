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
cfg = compose(config_name="config", overrides=["algorithm=ppo", "++algorithm.n_rollout_threads=1", "++algorithm.n_eval_rollout_threads=1"])
# Now cfg is DictConfig, set overrides via OmegaConf.update
OmegaConf.update(cfg, "subscriber.use_occlusion_features", False)
OmegaConf.update(cfg, "algorithm.use_lagrangian", True)
OmegaConf.update(cfg, "rew_cfg.near_miss_threshold", 2.0)
# Keep inactive agents True as runner does
cfg.subscriber.keep_inactive_agents = True
print("cfg max_num", cfg.max_num_vehicles, "occlusion", cfg.subscriber.use_occlusion_features)
from nocturne.envs.wrappers import create_ppo_env
from algos.ppo.env_wrappers import DummyVecEnv
def make_fn(rank=0):
    def _f():
        return create_ppo_env(cfg, rank=rank)
    return _f
env = create_ppo_env(cfg, rank=0)
print("env created")
# Check observation_space
obs_space = env.observation_space
print("obs_space type", type(obs_space), "len", len(obs_space) if isinstance(obs_space, list) else "single", "first shape", obs_space[0].shape if isinstance(obs_space, list) else obs_space.shape)
obs = env.reset()
print("reset type", type(obs), "len", len(obs) if isinstance(obs, list) else getattr(obs, 'shape', ''))
if isinstance(obs, list):
    shapes = [o.shape for o in obs]
    print("unique shapes", set(shapes))
    print("first 2", obs[0].shape, obs[1].shape if len(obs)>1 else "")
    dead = sum(1 for o in obs if np.all(o==-1))
    print("dead", dead)
    try:
        arr=np.array(obs)
        print("np.array shape", arr.shape)
    except Exception as e:
        print("np.array fail", e)
        for i,o in enumerate(obs):
            print(i, o.shape)

vec = DummyVecEnv([make_fn()])
print("vec created")
try:
    obs2 = vec.reset()
    print("vec obs2", type(obs2), getattr(obs2, 'shape', 'no shape'))
    if hasattr(obs2, 'shape'):
        print(obs2.shape)
    else:
        print("len", len(obs2))
except Exception as e:
    import traceback
    traceback.print_exc()

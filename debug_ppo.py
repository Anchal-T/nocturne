import os, sys
os.environ['DISPLAY']=':99'
os.environ['PROCESSED_TRAIN_NO_TL']='/home/anchal/anchal/nocturne/dataset/full_train/formatted_json_v2_no_tl_train'
sys.path.insert(0, '/home/anchal/anchal/nocturne')
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
initialize(config_path="cfgs", version_base=None)
cfg = compose(config_name="config", overrides=["algorithm=ppo", "++algorithm.n_rollout_threads=1", "++algorithm.n_eval_rollout_threads=1", "++subscriber.use_occlusion_features=False", "++algorithm.use_lagrangian=True", "++rew_cfg.near_miss_threshold=2.0"])
cfg = OmegaConf.to_container(cfg, resolve=True)
# Merge like runner does: for k,v in cfg['algorithm'].items(): cfg[k]=v
for k,v in cfg.get('algorithm', {}).items():
    cfg[k]=v
print("cfg max_num_vehicles", cfg['max_num_vehicles'], "subscriber occlusion", cfg['subscriber']['use_occlusion_features'])
from nocturne.envs.wrappers import create_ppo_env
from algos.ppo.env_wrappers import DummyVecEnv
def make_fn(rank=0):
    def _f():
        return create_ppo_env(cfg, rank=rank)
    return _f

# Test single env
env = create_ppo_env(cfg, rank=0)
print("single env obs space", type(env.observation_space), len(env.observation_space) if isinstance(env.observation_space, list) else env.observation_space)
obs = env.reset()
print("single env reset type", type(obs), "len", len(obs))
print("shapes", [o.shape for o in obs])
print("dtypes", [o.dtype for o in obs])
# check dead count
dead = sum(1 for o in obs if np.all(o==-1))
print("dead", dead)
# Check if all same shape
unique = set([o.shape for o in obs])
print("unique shapes", unique)
if len(unique)!=1:
    for i,o in enumerate(obs):
        print(i, o.shape, o[:5] if len(o)>5 else o)

# Now via DummyVecEnv
vec = DummyVecEnv([make_fn()])
print("vec created, calling reset")
try:
    obs2 = vec.reset()
    print("vec obs2 shape", obs2.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    # manual debug: get raw obs list
    obs_list = [env.reset() for env in vec.envs]
    print("manual obs_list len", len(obs_list))
    print("obs_list[0] len", len(obs_list[0]), "shapes", [o.shape for o in obs_list[0]][:5])

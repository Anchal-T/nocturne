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
cfg = OmegaConf.to_container(cfg, resolve=True)
for k,v in cfg.get('algorithm', {}).items():
    cfg[k]=v
print("cfg max_num", cfg['max_num_vehicles'], "occlusion", cfg['subscriber']['use_occlusion_features'])
from nocturne.envs.wrappers import create_ppo_env
from algos.ppo.env_wrappers import DummyVecEnv
def make_fn(rank=0):
    def _f():
        return create_ppo_env(cfg, rank=rank)
    return _f
env = create_ppo_env(cfg, rank=0)
print("env created, obs_space len", len(env.observation_space), "shape", env.observation_space[0].shape)
obs = env.reset()
print("reset type", type(obs), "len", len(obs))
print("shapes", [o.shape for o in obs])
print("unique", set([o.shape for o in obs]))
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
print("vec")
try:
    obs2 = vec.reset()
    print("vec obs2 shape", obs2.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    # manual
    obs_list = [env.reset() for env in vec.envs]
    print("manual len", len(obs_list), "first len", len(obs_list[0]))

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
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
for k,v in cfg_dict.get('algorithm', {}).items():
    cfg_dict[k]=v
# Bunch
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
cfg_b = Bunch(cfg_dict)
# also need to ensure cfg_b has .algo? The runner uses cfg.algorithm? Let's check wrapper expects cfg.img_as_state etc from top level
print("img_as_state", cfg_dict.get('img_as_state'))
print("max_num", cfg_dict.get('max_num_vehicles'))
from nocturne.envs.wrappers import create_ppo_env
from algos.ppo.env_wrappers import DummyVecEnv
def make_fn(rank=0):
    def _f():
        return create_ppo_env(cfg_b, rank=rank)
    return _f
env = create_ppo_env(cfg_b, rank=0)
print("env created", type(env), "obs space len", len(env.observation_space) if hasattr(env.observation_space, '__len__') else env.observation_space)
obs = env.reset()
print("reset len", len(obs), "type", type(obs))
print("shapes", [o.shape for o in obs][:5], "...", [o.shape for o in obs][-5:])
print("unique", set([o.shape for o in obs]))
dead = sum(1 for o in obs if np.all(o==-1))
print("dead", dead)
# Check np.array
try:
    arr=np.array(obs)
    print("np.array shape", arr.shape, arr.dtype)
except Exception as e:
    print("fail", e)
    for i,o in enumerate(obs):
        print(i, o.shape)

vec = DummyVecEnv([make_fn()])
print("vec")
obs2 = vec.reset()
print("vec obs2 shape", obs2.shape if hasattr(obs2, 'shape') else type(obs2))

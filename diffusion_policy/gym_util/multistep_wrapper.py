import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
        # Off by default. Image runners enable this only for explicit Zarr
        # exports, to retain each base-environment transition rather than just
        # the stacked observation seen at replanning boundaries.
        self._record_trajectory = False
        self._recorded_episode = None
        self._record_action_override = None
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        # obs_zeros = dict()
        # for key in obs.keys():
        #     obs_zeros[key] = np.zeros_like(obs[key])
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
        if self._record_trajectory:
            self._recorded_episode = {
                "obs": {key: [] for key in obs},
                "action": [],
                "reward": [],
                "done": [],
            }
        self._record_action_override = None

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for action_index, act in enumerate(action):
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            if self._record_trajectory:
                if self._recorded_episode is None:
                    raise RuntimeError("Recording was enabled after reset.")
                for key, value in self.obs[-1].items():
                    self._recorded_episode["obs"][key].append(
                        np.asarray(value).copy())
                recorded_action = act
                if self._record_action_override is not None:
                    recorded_action = self._record_action_override[action_index]
            observation, reward, done, info = super().step(act)

            if self._record_trajectory:
                self._recorded_episode["action"].append(
                    np.asarray(recorded_action).copy())
                self._recorded_episode["reward"].append(
                    np.asarray(reward, dtype=np.float32))
                self._recorded_episode["done"].append(
                    np.asarray(done, dtype=np.bool_))

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def start_recording(self):
        """Begin raw-step capture on the next reset."""
        self._record_trajectory = True
        self._recorded_episode = None
        self._record_action_override = None

    def set_recording_actions(self, action):
        """Save policy-space actions when environment controls are transformed."""
        if not self._record_trajectory:
            return
        action = np.asarray(action)
        if action.ndim < 1:
            raise ValueError("Recorded action chunk must have a time dimension.")
        self._record_action_override = action

    def get_recorded_episode(self, clear=True):
        """Return a ReplayBuffer-compatible raw-step episode."""
        episode = self._recorded_episode
        if episode is None or not episode["action"]:
            return None
        result = {key: np.stack(values) for key, values in episode["obs"].items()}
        result.update({
            "action": np.stack(episode["action"]),
            "reward": np.asarray(episode["reward"], dtype=np.float32),
            "done": np.asarray(episode["done"], dtype=np.bool_),
        })
        if clear:
            self._recorded_episode = None
        return result


    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

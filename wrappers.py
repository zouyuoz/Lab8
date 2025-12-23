import numpy as np
import gymnasium as gym
import torchvision.transforms as T
from stable_baselines3.common.monitor import Monitor
from collections import deque
import cv2

# 要改
class PreprocessObsWrapper(gym.ObservationWrapper):
    """Fixed preprocessing: resize to 224, normalize to [-1,1], output CHW float32."""

    def __init__(self, env):
        super().__init__(env)
        # obs_space = env.observation_space
        # if not isinstance(obs_space, gym.spaces.Box):
        #     raise TypeError("PreprocessObsWrapper requires a Box observation space")
        self.resize_shape = (84, 84)
        # c = obs_space.shape[2]
        c = 1
        low  = np.full((c, 84, 84), -1.0, dtype=np.float32)
        high = np.full((c, 84, 84),  1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # transforms = []
        # transforms.append(T.ToTensor())  # -> float CHW in [0,1]
        # transforms.append(T.Resize(self.resize_shape, antialias=True))
        # transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        # transforms.append(T.Lambda(lambda x: x.numpy()))
        # self.pipeline = T.Compose(transforms)

    def observation(self, observation):
        # 1. Resize (使用 INTER_AREA 縮小效果最好且快)
        # resized = cv2.resize(observation, self.resize_shape, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=2)

        # 2. Normalize & Transpose
        # (H, W, C) -> (C, H, W) 並正規化到 [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return normalized.transpose(2, 0, 1)


# 應該不需要改
class DiscreteActionWrapper(gym.ActionWrapper):
    """
    change action space from MultiBinary to Discrete with predefined button combos
    """
    def __init__(self, env, combos):
        super().__init__(env)


        if not hasattr(env.unwrapped, "buttons"):
            raise ValueError("unsupported env, must have 'buttons' attribute")

        self.buttons = list(env.unwrapped.buttons)  # e.g. ['B','Y','SELECT',...]
        self.button_to_idx = {b: i for i, b in enumerate(self.buttons)}

        # Get combos
        self.combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

        self._mapped = []
        n = env.action_space.n  # MultiBinary(n)
        for keys in combos:
            a = np.zeros(n, dtype=np.int8)
            for k in keys:
                if k not in self.button_to_idx:
                    raise ValueError(f"unsupported buttons in this env.buttons: {self.buttons}")
                a[self.button_to_idx[k]] = 1
            self._mapped.append(a)

    def action(self, act):
        return self._mapped[int(act)].copy()

# 死掉的話要重製
class LifeTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def _get_lives(self, info):
        if not isinstance(info, dict):
            return None
        if "lives" in info:
            return int(info["lives"])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lives = self._get_lives(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = self._get_lives(info)

        died = False
        if lives is not None and self._prev_lives is not None:
            if lives < self._prev_lives:
                died = True
        self._prev_lives = lives

        if died:
            terminated = True
            if isinstance(info, dict):
                info = dict(info)
                info["death"] = True

        if info.get("is_cleared", False):
            terminated = True

        return obs, reward, terminated, truncated, info

# 遊戲 info (金幣、分數) 不用動
class ExtraInfoWrapper(gym.Wrapper):
    """
    Attach extra RAM-derived signals (HUD timer, x-position) to info.
    """

    TIMER_HUNDREDS = 0x0F31
    TIMER_TENS = 0x0F32
    TIMER_ONES = 0x0F33
    # In SMW RAM, $0094 stores the low byte and $0095 stores the high byte.
    X_POS_LOW  = 0x0094
    X_POS_HIGH = 0x0095
    # Add Y_POS_LOW and Y_POS_HIGH
    Y_POS_LOW  = 0x0096
    Y_POS_HIGH = 0x0097

    GAME_MODE = 0x0100
    ANIMATION = 0x0071

    def __init__(self, env):
        super().__init__(env)
        self._episode_start_x = None

    def _get_ram(self):
        base_env = self.env.unwrapped
        if not hasattr(base_env, "get_ram"):
            return None
        return base_env.get_ram()

    def _read_time_left(self, ram):
        if ram is None:
            return None
        hundreds = int(ram[self.TIMER_HUNDREDS]) & 0x0F
        tens = int(ram[self.TIMER_TENS]) & 0x0F
        ones = int(ram[self.TIMER_ONES]) & 0x0F
        return hundreds * 100 + tens * 10 + ones

    def _read_x_pos(self, ram):
        if ram is None:
            return None
        low = int(ram[self.X_POS_LOW])
        high = int(ram[self.X_POS_HIGH])
        return (high << 8) | low

    def _read_y_pos(self, ram):
        if ram is None:
            return None
        low = int(ram[self.Y_POS_LOW])
        high = int(ram[self.Y_POS_HIGH])
        return (high << 8) | low

    def _read_game_mode(self, ram):
        if ram is None: return None
        return int(ram[self.GAME_MODE])

    def _read_animation(self, ram):
        if ram is None: return None
        return int(ram[self.ANIMATION])

    def _inject_extra(self, info):
        ram       = self._get_ram()
        time_left = self._read_time_left(ram)
        x_pos     = self._read_x_pos(ram)
        y_pos     = self._read_y_pos(ram)
        game_mode = self._read_game_mode(ram)
        anime     = self._read_animation(ram)

        if time_left is None and x_pos is None:
            return info
        if not isinstance(info, dict):
            info = {}

        # copy to avoid mutating shared dict instances
        info = dict(info)
        if time_left is not None:
            info["time_left"] = time_left
        if x_pos is not None:
            if self._episode_start_x is None:
                self._episode_start_x = x_pos
            info["x_pos"] = max(0, x_pos - self._episode_start_x)
        if y_pos is not None:
            info["y_pos"] = y_pos
        if game_mode is not None:
            info["game_mode"] = game_mode
            info["is_cleared"] = (game_mode == 14)
        if anime is not None:
            info["anime"] = anime
            info["pipe"] = (anime == 5 or anime == 6)

        return info

    def reset(self, **kwargs):
        self._episode_start_x = None
        obs, info = self.env.reset(**kwargs)
        info = self._inject_extra(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_extra(info)
        return obs, reward, terminated, truncated, info

# 除了回傳 image (OBS) 之外還需要哪些資訊
# 現在有 step 根所剩時間
# 可以改
class AuxObservationWrapper(gym.Wrapper):
    """
    Convert image observations into a dict that also exposes scalar features (step/time).
    """

    def __init__(self, env, step_normalizer: float = 18000.0, time_normalizer: float = 300.0):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("AuxObservationWrapper expects a Box observation space as the image input")
        self.image_space = env.observation_space
        self.step_normalizer = max(step_normalizer, 1.0)
        self.time_normalizer = max(time_normalizer, 1.0)
        scalar_low = np.full((2,), -np.inf, dtype=np.float32)
        scalar_high = np.full((2,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": self.image_space,
                "scalars": gym.spaces.Box(low=scalar_low, high=scalar_high, dtype=np.float32),
            }
        )
        self._step_count = 0

    def _make_obs(self, obs, info):
        time_left = float(info.get("time_left", 0.0)) if isinstance(info, dict) else 0.0
        time_feat = np.clip(time_left / self.time_normalizer, 0.0, 1.0)
        step_feat = np.clip(self._step_count / self.step_normalizer, 0.0, 1.0)
        scalars = np.array([step_feat, time_feat], dtype=np.float32)
        return {"image": obs, "scalars": scalars}

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return self._make_obs(obs, info), reward, terminated, truncated, info

# 最最重要!!!!!!!!!!!!!!!!!!!!!!
# 自己去實驗
class RewardOverrideWrapper(gym.Wrapper):
    """
    Replace environment reward with custom shaping
    """

    def __init__(
        self,
        env,
        win_reward: float = 2.5,
    ):
        super().__init__(env)
        self.win_reward = win_reward
        self._prev_score = None
        self._prev_x = None
        self._prev_y = None
        # self._prev_time = None
        self._prev_coin = None
        self.max_x = 0
        self.stuck_counter = 0 # add penalty when stuck in wall
        self.gate_remained = 2 # there are two blocks

    def _reset_trackers(self, info):
        self._prev_score = info.get("score", 0)
        self._prev_x = info.get("x_pos", 0)
        self._prev_y = info.get("y_pos", 315)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        self._reset_trackers(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        # if win, imidiately return
        if info.get("is_cleard", False):
            return obs, self.win_reward, True, truncated, info

        reward = 0.0
        # Distance reward
        x_pos = info.get("x_pos", 0)
        dx = x_pos - self._prev_x

        if dx != 0: self.stuck_counter = 0
        if dx > 0:
            reward += dx * 0.02
        if x_pos > self.max_x:
            self.max_x = x_pos
        self._prev_x = x_pos

        # Encourage agent to jump
        # Gain higher score when y_pos is low (high in obs)
        y_pos = info.get("y_pos", 0)
        dy = y_pos - self._prev_y
        if dy < 0:
            reward += 1 / max(y_pos, 125)
            self.stuck_counter = 0
        self._prev_y = y_pos

        # Stuck Penalty
        action_bad = [0, 6]
        if dx == 0 and dy == 0: self.stuck_counter += 1
        if self.stuck_counter > 25 and action in action_bad:
            reward -= 0.02

        # Time Penalty
        reward -= 0.01

        # Reward for score increments
        score = info.get("score", 0)
        if self._prev_score is None:
            self._prev_score = score
        else:
            dScore = score - self._prev_score # 5, 10, 20, 40, 80, 100
            if dScore > 0:
                if dScore == 5 and (1850 < x_pos < 2000): # distroy secret tunnel surface
                    self.gate_remained -= 1
                    reward += 10
                else:
                    reward += 0.1 * dScore
                self._prev_score = score

        # Secret tunnel
        is_in_pipe = info.get("pipe", False)
        spin_jumps= [4, 6, 8] # "A"s, "8 is now leftA"
        destroying_gate = (
            (1900 < x_pos < 1930) and
            ( 280 < y_pos <  295) and
            dy != 0 and
        	action in spin_jumps and
			self.gate_remained != 0
   		)
        if destroying_gate:
            reward += 1
        if (1910 < x_pos < 1920) and y_pos > 310 and is_in_pipe: # squat (action == 3)
            reward += 2

        coin = info.get("coins", 0)
        if self._prev_coin is None:
            self._prev_coin = coin
        else:
            dCoin = coin - self._prev_coin
            reward += dCoin # usually increase by 1
            self._prev_coin = coin

        # Death & Win Handling
        if info.get("death", False):
            # reward -= 1.0
            reward = reward - 1.0 if reward < 10 else reward * 0.9
        elif terminated and not truncated:
            reward += self.win_reward

        time_left = info.get("time_left")
        is_cleared = info.get("is_cleared", False)
        if time_left == 0 and not is_cleared:
            reward = -1.0
            terminated = True

        if terminated or truncated:
            self._reset_trackers(info)

        return obs, reward, terminated, truncated, info

class InfoLogger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0 or terminated:
            print(info)
        return obs, reward, terminated, truncated, info

class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, terminated, truncated = None, False, False
        info = {}

        # 讓遊戲引擎連續跑 skip 次
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc

            # 如果中間死掉了或遊戲結束，就提早跳出
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

class FrameStackWrapper(gym.Wrapper):
    """
    將連續的 n 幀畫面堆疊在一起，增加動態資訊。
    """
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)

        # 更新 observation_space 的維度 (C*n, H, W)
        low  = np.repeat(env.observation_space.low , n_frames, axis=0)
        high = np.repeat(env.observation_space.high, n_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # 在通道維度 (axis=0) 進行拼接
        return np.concatenate(list(self.frames), axis=0)

COMBOS = [
    [],             # 00: NOOP
    ["RIGHT"],      # 01: 走右
    ["LEFT"],       # 02: 走左（可選）
    ["DOWN"],       # 03: 下蹲
    ["A"],          # 04: 旋跳 (Spin Jump)
    ["B"],          # 05: 跳 (Jump)
    ["RIGHT", "A"], # 06: 右 + 旋跳
    ["RIGHT", "B"], # 07: 右 + 跳
    # ["RIGHT", "Y"], # 08: 右 + 跑
    ["LEFT", "A"],  # 09: 左 + 旋跳
    ["LEFT", "B"],  # 10: 左 + 跳
    # ["LEFT", "Y"],  # 11: 左 + 跑
    # ["Y"],          # 12: 加速 (在這關不會單獨使用)
]

import retro

def make_base_env(game: str, state: str):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = SkipFrameWrapper(env, skip=4) # 先加入 skip frame
    env = Monitor(env) # 記錄到的 step 會是 1/4
    env = PreprocessObsWrapper(env) # 主要是 resize
    env = FrameStackWrapper(env, n_frames=4) # 做 stack，已經是 skip-framed 畫面
    env = DiscreteActionWrapper(env, COMBOS)
    env = ExtraInfoWrapper(env)
    env = LifeTerminationWrapper(env)
    env = RewardOverrideWrapper(env)
    env = AuxObservationWrapper(env)
    return env

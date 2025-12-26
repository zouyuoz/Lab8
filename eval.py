import os
import argparse
from typing import Tuple
import numpy as np
from custom_policy import CustomPPO
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from wrappers import make_base_env, COMBOS
def evaluate_policy(model: CustomPPO, game: str, state: str, n_episodes: int, max_steps: int):
    env = make_base_env(game, state)
    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            steps += 1

        returns.append(ep_ret)

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    best_ret = float(np.max(returns)) if returns else 0.0
    return mean_ret, best_ret

def _format_info(info: dict, max_len: int = 48) -> str:
    if not isinstance(info, dict) or not info:
        return "{}"
    lines = []
    current_line = []
    current_line_len = 0
    for key, value in info.items():
        if key == "x_pos": key = "X"
        if key == "y_pos": key = "Y"
        if key == "time_left": key = "CLK"
        if key == "game_mode": key = "GM"
        if key == "is_cleared": continue
        if key == "lives": continue
        if key == "inter_frames": continue
        # if key == "sprites": continue

        fragment = f"{key}={value}"
        separator_len = 3 if current_line else 0
        if current_line_len + separator_len + len(fragment) > max_len:
            lines.append(" | ".join(current_line) + " |")
            current_line = [fragment]
            current_line_len = len(fragment)
        else:
            current_line.append(fragment)
            current_line_len += separator_len + len(fragment)
    if current_line:
        lines.append(" | ".join(current_line) + " |")
    return "\n".join(lines)


def _annotate_frame(frame: np.ndarray, cumulative_reward: float, last_reward: float, action: int, info: dict, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw_overlay = ImageDraw.Draw(overlay)

    action_label = COMBOS[action] if action < len(COMBOS) else "UNKNOWN"
    info_str = _format_info(info)
    lines = [
        f"RWD={last_reward:.3f} | C_RWD={cumulative_reward:.3f} | ACT={action},{action_label}",
        f"{info_str}",
        f"",
    ]
    padding = 4
    bbox_sample = draw_overlay.textbbox((0, 0), "Ag", font=font)
    line_height = bbox_sample[3] - bbox_sample[1]
    # line_widths = []
    # for line in lines:
    #     bbox = draw_overlay.textbbox((0, 0), line, font=font)
    #     line_widths.append(bbox[2] - bbox[0])
    # box_width = max(line_widths) + padding * 2
    box_width = 256
    box_height = line_height * len(lines) + padding * (len(lines) + 1)
    # box_height = line_height * 4 + padding * (4 + 1)
    draw_overlay.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0, 160))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 196, 0), font=font)
        y += line_height + padding
    return np.array(img.convert("RGB"))


def record_video(model: CustomPPO, game: str, state: str, out_dir: str, video_len: int, prefix: str, record: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.mp4")

    env = make_base_env(game, state, record=record)
    fps = env.metadata.get("render_fps", 60)
    writer = imageio.get_writer(out_path, fps=fps)
    font = ImageFont.load_default()

    obs, info = env.reset()
    cumulative_reward = 0.0
    for _ in range(video_len):
        action, _ = model.predict(obs, deterministic=True)
        act_val = int(action) if hasattr(action, "__init__") else action
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)

        raw_frames = info.get("inter_frames", [env.render()])
        for frame in raw_frames:
            if frame is None: continue
            annotated = _annotate_frame(frame, cumulative_reward, float(reward), act_val, info, font)
            writer.append_data(annotated)

        if terminated or truncated:
            obs, info = env.reset()
            cumulative_reward = 0.0

    writer.close()
    env.close()
    print(f"Saved video to {out_path}")
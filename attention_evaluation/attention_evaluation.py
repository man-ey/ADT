import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.atari_wrappers import WarpFrame, AtariWrapper, FireResetEnv, NoopResetEnv,MaxAndSkipEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
from ocatari.core import OCAtari
from attention_utils import AtariDQNWrapper, make_env_for_ckpt, load_atarinet

#GAZE_WT = "/home/arch/Documents/gitRepos/humangaze/code/human_gaze_models/breakout.hdf5"            # Zhang et al. network weights
#MEAN_NPZ = "/home/arch/Documents/gitRepos/humangaze/code/human_gaze_models/breakout.mean.npy"        # per-pixel training mean
GAME = "Freeway"

MODEL_PATH = "resources/dqn_checkpoints/"+GAME+"/0/model_50000000.gz"
ENV_ID = GAME + "NoFrameskip-v4"
NUM_STEPS = 1000
THRESHOLD_RATIO = 0.3  # top 30% of salient pixels
N_ENVS = 1

RAW_H, RAW_W = 210, 160  # Typical Atari raw frame size (height=210, width=160)

def create_env_old():
    """
    Creates a vectorized Atari environment wrapped to produce
    84x84 (grayscale) frames, transposed to (channel, height, width),
    then stacked (4 frames).
    """
    def make_base():                                  # this fn is called once per env
        base = OCAtari(
            ENV_ID,                 # "PongNoFrameskip-v4"
            mode="vision",          # keep pixel observations; objects via colour masks
            render_mode="rgb_array" # so env.render() still works
        )
        base = WarpFrame(base)                       # 84×84 gray
        return base
    env = make_vec_env(make_base, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env

def create_env(game_id=ENV_ID):
    """
Returns a 1-env DummyVecEnv whose behaviour is identical to ALEClassic:
• frame_skip = 4           (repeat each chosen action 4 times)
• sticky_action_p = 0.0    (no stochasticity, like Nature-DQN)
• terminal_on_life_loss = False
Obs shape: (1, 4, 84, 84) uint8  — same as training.
    """
    def _make():
        env = OCAtari(
            game_id,
            mode="vision",
            obs_mode="dqn",
            render_mode="rgb_array",
            frameskip=4,                  # deterministic 4-step repeat
            repeat_action_probability=0.0,
            )
        # Wrap gymnasium environment for compatibility reasons for games with action fireactionreset works as well
        env = MaxAndSkipEnv(env, skip=1)
        return env


    return DummyVecEnv([_make])

def detect_dynamic_objects(current_frame, prev_frame, min_size=30, diff_thresh=20,
                           expand=15):
    diff = cv2.absdiff(current_frame, prev_frame)  # (210,160,3)
    diff_gray = diff.max(axis=2)                  # (210,160)

    # Threshold
    moving_mask = (diff_gray > diff_thresh).astype(np.uint8)

    # Optional morphological close
    kernel = np.ones((3,3), np.uint8)
    moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels = cv2.connectedComponents(moving_mask)
    bboxes = []
    h, w = diff_gray.shape  # h=210, w=160
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) < min_size:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # -------------------------
        # Expand bounding box
        # -------------------------
        x1 = max(0, x1 - expand)
        x2 = min(w - 1, x2 + expand)
        y1 = max(0, y1 - expand)
        y2 = min(h - 1, y2 + expand)

        bboxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return bboxes

def bboxes_from_ocatari(obj_list):
    """Convert OCAtari objects -> dicts compatible with your utilities."""
    boxes = []
    for o in obj_list:
        x1, y1 = int(o.x), int(o.y)
        x2, y2 = x1 + int(o.w) - 1, y1 + int(o.h) - 1
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def compute_saliency_for_rgb(model, obs_frame):

    frame_tensor = torch.FloatTensor(obs_frame.transpose(2, 0, 1))
    frame_tensor = frame_tensor.unsqueeze(0).to(model.device)
    frame_tensor.requires_grad = True

    with torch.enable_grad():
        features = model.policy.extract_features(frame_tensor)
        policy_latent, value_latent = model.policy.mlp_extractor(features)
        policy_logits = model.policy.action_net(policy_latent)
        chosen_action = torch.argmax(policy_logits, dim=1)
        chosen_logit = policy_logits[0, chosen_action]
        chosen_logit.backward()

    grad = frame_tensor.grad.abs().squeeze(0)   # -> (C, 84, 84)
    saliency_84x84 = grad.max(dim=0)[0]         # -> (84, 84)
    return saliency_84x84.cpu().numpy()

def compute_saliency_for_rgb_dqn(model, obs_frame_4x84x84):
    """
obs_frame_4x84x84: numpy array of shape (4, 84, 84), stacked frames
    """
    frame_tensor = torch.FloatTensor(obs_frame_4x84x84).unsqueeze(0).to(model.device)
    frame_tensor.requires_grad = True

    with torch.enable_grad():
        # Get Q-values
        latent = model.q_net.features_extractor(frame_tensor)
        q_values = model.q_net.q_net(latent)
        # Pick the argmax action
        chosen_action = torch.argmax(q_values, dim=1)
        chosen_q_value = q_values[0, chosen_action]
        # Backprop
        chosen_q_value.backward()

    # Gradient w.r.t. input frames
    grad = frame_tensor.grad.abs().squeeze(0)  # (4, 84, 84)
    # Take max across the 4 channels (the stacked frames)
    saliency_84x84 = grad.max(dim=0)[0]  # -> shape (84, 84)
    return saliency_84x84.cpu().numpy()

def compute_saliency_for_rgb_dqn_net(model, obs_frame_4x84x84):
    """
SB3-free version that works with AtariDQNWrapper.
    """
    frame = torch.as_tensor(obs_frame_4x84x84, dtype=torch.float32,
                            device=model.device).unsqueeze(0)
    frame.requires_grad_()

    q_vals = model.net(frame)              # direct forward
    best = q_vals.argmax(1)
    q_vals[0, best].backward()

    grad = frame.grad.abs().squeeze(0)     # (4,84,84)
    return grad.max(0)[0].cpu().numpy()    # (84,84)


def keep_topk(arr2d: np.ndarray, ratio: float) -> np.ndarray:
    """Zero out everything except the top-k% values."""
    if arr2d.size == 0:
        return arr2d
    k = max(1, int(arr2d.size * ratio))
    thresh = np.partition(arr2d.ravel(), -k)[-k]
    masked = np.where(arr2d >= thresh, arr2d, 0.0)
    return masked


def upsample_saliency(saliency_84x84, target_h=210, target_w=160):
    """
Resizes saliency from (84,84) -> (210,160) using nearest-neighbor.
    """
    up_sal = cv2.resize(
        saliency_84x84,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    return up_sal

def measure_saliency_in_bboxes_old(saliency_2D, bboxes, threshold_ratio=THRESHOLD_RATIO):

    flat = saliency_2D.ravel()
    if flat.size == 0 or len(bboxes) == 0:
        return 0.0  # If no bounding boxes, return 0.0 overlap

    # 1) Identify the saliency cutoff for top fraction
    sorted_vals = np.sort(flat)[::-1]  # descending
    cutoff_idx = int(len(sorted_vals) * threshold_ratio)
    cutoff_idx = max(cutoff_idx, 1)
    thresh_val = sorted_vals[cutoff_idx - 1]

    # 2) Build a mask of "salient" pixels above that threshold
    salient_mask = (saliency_2D >= thresh_val)
    total_salient = np.count_nonzero(salient_mask)
    if total_salient == 0:
        return 0.0

    # 3) Count how many of those salient pixels are inside any bounding box
    ys, xs = np.where(salient_mask)
    inside = 0
    for y, x in zip(ys, xs):
        in_box = any(
            (x >= box['x1'] and x <= box['x2'] and
             y >= box['y1'] and y <= box['y2'])
            for box in bboxes
        )
        if in_box:
            inside += 1

    return inside / total_salient

def measure_saliency_in_bboxes(
        saliency_2D,
        bboxes,
        threshold_ratio=0.3,
        radius=30           # <-- new
):
    h, w = saliency_2D.shape
    if saliency_2D.size == 0 or len(bboxes) == 0:
        return 0.0

    # 1) expand each box by ±radius, but keep it inside the frame
    pad_boxes = []
    for b in bboxes:
        pad_boxes.append({
            "x1": max(0, b["x1"] - radius),
            "y1": max(0, b["y1"] - radius),
            "x2": min(w - 1, b["x2"] + radius),
            "y2": min(h - 1, b["y2"] + radius),
        })

    # 2) pick the top-k % salient pixels, same as before
    flat   = saliency_2D.ravel()
    k      = max(1, int(len(flat) * threshold_ratio))
    thresh = np.partition(flat, -k)[-k]          # kth-largest value
    salient_mask = saliency_2D >= thresh
    if not salient_mask.any():
        return 0.0

    # 3) count how many salient pixels fall inside at least one padded box
    ys, xs = np.nonzero(salient_mask)
    inside = sum(
        any(b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"] for b in pad_boxes)
        for y, x in zip(ys, xs)
    )

    return inside / salient_mask.sum()


def overlay_saliency_on_frame(raw_frame_210x160x3, saliency_210x160, alpha=0.5):
    """
Overlays a grayscale saliency map on an RGB frame by coloring it (JET colormap)
and blending with the original image.
Returns shape (210,160,3) in RGB.
    """
    frame_bgr = cv2.cvtColor(raw_frame_210x160x3, cv2.COLOR_RGB2BGR)

    # Normalize saliency to 0..255
    sal = saliency_210x160.astype(np.float32)
    if sal.size > 0:
        sal -= sal.min()
        maxv = sal.max() + 1e-9
        sal = sal / maxv
    sal = (sal * 255.0).astype(np.uint8)

    # Apply JET colormap
    sal_colored_bgr = cv2.applyColorMap(sal, cv2.COLORMAP_JET)

    # Blend images
    blended_bgr = cv2.addWeighted(sal_colored_bgr, alpha, frame_bgr, 1 - alpha, 0)
    # Convert back to RGB for matplotlib
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    return blended_rgb

def evaluate_attention_dqn(model, env, num_steps=NUM_STEPS, do_render=True):
    obs = env.reset()
    prev_raw_frame = None

    # Setup live rendering
    if do_render:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im_left = ax[0].imshow(np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8))
        im_right = ax[1].imshow(np.zeros((RAW_H, RAW_W)), cmap='hot', vmin=0, vmax=1)
        ax[0].set_title("Raw Frame + Saliency Overlay")
        ax[1].set_title("Saliency (Upsampled)")
        plt.tight_layout()

    total_overlap = 0.0
    step_count = 0

    for step in range(num_steps):
        # 1) Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs_new, rewards, dones, infos = env.step(action)

        # 2) current observation for saliency: (4, 84, 84)
        agent_frame_4x84x84 = obs_new[0]

        # 3) Render the current raw frame
        current_raw_frame = env.render(mode='rgb_array')  # shape (210,160,3)

        # 4) Compute saliency on the agent's 4x84x84
        sal_84x84 = compute_saliency_for_rgb_dqn(model, agent_frame_4x84x84)

        # 5) Upsample saliency to (210,160)
        sal_210x160 = upsample_saliency(sal_84x84, target_h=RAW_H, target_w=RAW_W)

        # 6) previous frame, detect dynamic objects
        if prev_raw_frame is not None:
            bboxes = detect_dynamic_objects(current_raw_frame, prev_raw_frame,
                                            min_size=30, diff_thresh=20)

            # Only measure overlap *if* found at least one bounding box
            if len(bboxes) > 0:
                overlap = measure_saliency_in_bboxes(sal_210x160, bboxes, threshold_ratio=THRESHOLD_RATIO)
                total_overlap += overlap
                step_count += 1
            else:
                # If no objects found,not increment step_count
                overlap = 0.0

            # Visualization
            if do_render:
                overlay_img = overlay_saliency_on_frame(current_raw_frame, sal_210x160, alpha=0.5)
                im_left.set_data(overlay_img)

                sal_top = keep_topk(sal_210x160, THRESHOLD_RATIO)  # keep only top 30%
                im_right.set_data(sal_top / (sal_top.max() + 1e-9))
                im_right.set_clim(0, 1)

                ax[0].set_title(f"Overlay (step={step}), #Objects={len(bboxes)}")
                ax[1].set_title(f"Saliency Overlap={overlap:.2f}")
                plt.pause(0.01)

        prev_raw_frame = None if dones[0] else current_raw_frame

        obs = obs_new
        if dones[0]:
            obs = env.reset()

    avg_overlap = total_overlap / step_count if step_count > 0 else 0.0
    print(f"Average fraction of top-salient pixels in moving objects = {avg_overlap:.3f}")

def evaluate_attention(model, env, num_steps=NUM_STEPS, do_render=True):
    obs = env.reset()
    prev_raw_frame = None

    # Setup live rendering
    if do_render:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im_left = ax[0].imshow(np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8))
        im_right = ax[1].imshow(np.zeros((RAW_H, RAW_W)), cmap='hot', vmin=0, vmax=1)
        ax[0].set_title("Raw Frame + Saliency Overlay")
        ax[1].set_title("Saliency (Upsampled)")
        plt.tight_layout()

    total_overlap = 0.0
    step_count = 0

    for step in range(num_steps):
        # 1) Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs_new, rewards, dones, infos = env.step(action)

        # 2) Our current observation for saliency: (4, 84, 84)
        agent_frame_4x84x84 = obs_new[0]

        # 3) Render the current raw frame
        current_raw_frame = env.render(mode='rgb_array')  # shape (210,160,3)

        # 4) Compute saliency on the agent's 4x84x84
        sal_84x84 = compute_saliency_for_rgb(model, agent_frame_4x84x84)

        # 5) Upsample saliency to (210,160)
        sal_210x160 = upsample_saliency(sal_84x84, target_h=RAW_H, target_w=RAW_W)

        # 6) If previous frame, detect dynamic objects
        if prev_raw_frame is not None:
            bboxes = detect_dynamic_objects(current_raw_frame, prev_raw_frame,
                                            min_size=5, diff_thresh=10)

            # Only measure overlap *if* found at least one bounding box
            if len(bboxes) > 0:
                overlap = measure_saliency_in_bboxes(sal_210x160, bboxes, threshold_ratio=THRESHOLD_RATIO)
                total_overlap += overlap
                step_count += 1
            else:
                # If no objects found, not increment step_count
                overlap = 0.0

            # Visualization
            if do_render:
                overlay_img = overlay_saliency_on_frame(current_raw_frame, sal_210x160, alpha=0.5)
                im_left.set_data(overlay_img)

                sal_top = keep_topk(sal_210x160, THRESHOLD_RATIO)  # keep only top 30%
                im_right.set_data(sal_top / (sal_top.max() + 1e-9))
                im_right.set_clim(0, 1)

                ax[0].set_title(f"Overlay (step={step}), #Objects={len(bboxes)}")
                ax[1].set_title(f"Saliency Overlap={overlap:.2f}")
                plt.pause(0.01)

        prev_raw_frame = None if dones[0] else current_raw_frame

        obs = obs_new
        if dones[0]:
            obs = env.reset()

    #  Final average overlap
    #    only steps that actually had bounding boxes are counted.
    avg_overlap = total_overlap / step_count if step_count > 0 else 0.0
    print(f"Average fraction of top-salient pixels in moving objects = {avg_overlap:.3f}")

def overlay_saliency_on_frame_topk(raw_rgb, saliency_210x160, ratio=0.3, alpha=0.5):
    frame_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)

    # build mask + normalize only within mask
    k = max(1, int(saliency_210x160.size * ratio))
    thresh = np.partition(saliency_210x160.ravel(), -k)[-k]
    mask = saliency_210x160 >= thresh

    norm = np.zeros_like(saliency_210x160, dtype=np.float32)
    if mask.any():
        vals = saliency_210x160[mask].astype(np.float32)
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        norm[mask] = vals

    sal8 = (norm * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(sal8, cv2.COLORMAP_JET)

    blended = frame_bgr.copy()
    blended_masked = cv2.addWeighted(color_bgr, alpha, frame_bgr, 1 - alpha, 0)
    blended[mask] = blended_masked[mask]
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def draw_bboxes_on_image(image_rgb, bboxes, color=(255,255,255), thickness=3):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for box in bboxes:
        cv2.rectangle(
            image_bgr,
            (box['x1'], box['y1']),  # top-left
            (box['x2'], box['y2']),  # bottom-right
            color,
            thickness
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def evaluate_attention_draw_box(model, env, num_steps=NUM_STEPS, do_render=True):
    obs = env.reset()
    prev_raw_frame = None

    if do_render:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im_left = ax[0].imshow(np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8))
        im_right = ax[1].imshow(np.zeros((RAW_H, RAW_W)), cmap='hot', vmin=0, vmax=1)
        ax[0].set_title("Raw Frame + Saliency Overlay + BBoxes")
        ax[1].set_title("Saliency (Upsampled)")
        plt.tight_layout()

    total_overlap = 0.0
    step_count = 0

    for step in range(num_steps):
        # 1) Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs_new, rewards, dones, infos = env.step(action)

        # 2) Current observation for saliency: (4, 84, 84)
        agent_frame_4x84x84 = obs_new[0]

        # 3) Render the current raw frame
        current_raw_frame = env.render(mode='rgb_array')  # shape (210,160,3)

        # 4) Compute saliency
        sal_84x84 = compute_saliency_for_rgb(model, agent_frame_4x84x84)

        # 5) Upsample saliency to (210,160)
        sal_210x160 = upsample_saliency(sal_84x84, target_h=RAW_H, target_w=RAW_W)

        # 6) If we have a previous frame, detect dynamic objects
        if prev_raw_frame is not None:
            bboxes = detect_dynamic_objects(current_raw_frame, prev_raw_frame,
                                            min_size=5, diff_thresh=5)
            # 7) Measure overlap if we found any bounding boxes
            if len(bboxes) > 0:
                overlap = measure_saliency_in_bboxes(sal_210x160, bboxes, threshold_ratio=THRESHOLD_RATIO)
                total_overlap += overlap
                step_count += 1
            else:
                overlap = 0.0

            # 8) Visualization
            if do_render:
                frame_with_bboxes = draw_bboxes_on_image(current_raw_frame, bboxes)
                im_left.set_data(frame_with_bboxes)

                # Show the raw upsampled saliency on the right
                sal_top = keep_topk(sal_210x160, THRESHOLD_RATIO)  # keep only top 30%
                im_right.set_data(sal_top / (sal_top.max() + 1e-9))
                im_right.set_clim(0, 1)

                ax[0].set_title(f"Overlay+BBoxes (step={step}), #Objects={len(bboxes)}")
                ax[1].set_title(f"Saliency Overlap={overlap:.2f}")
                plt.pause(0.01)

        # Update prev_raw_frame
        prev_raw_frame = None if dones[0] else current_raw_frame

        obs = obs_new
        if dones[0]:
            obs = env.reset()

    avg_overlap = total_overlap / step_count if step_count > 0 else 0.0
    print(f"Average fraction of top-salient pixels in moving objects = {avg_overlap:.3f}")

def evaluate_attention_dqn_ocatari(model, env, radius, num_steps=NUM_STEPS, do_render=True):
    obs = env.reset()
    total_overlap, step_count = 0.0, 0

    # live display
    if do_render:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im_left = ax[0].imshow(np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8))
        im_right = ax[1].imshow(np.zeros((RAW_H, RAW_W)), cmap='hot', vmin=0, vmax=1)
        ax[0].set_title("Raw Frame + Saliency Overlay + OCAtari boxes")
        ax[1].set_title("Saliency (Upsampled)")
        plt.tight_layout()

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # 1) saliency exactly as before
        sal_84x84  = compute_saliency_for_rgb_dqn_net(model, obs[0])
        sal_210x160 = upsample_saliency(sal_84x84, target_h=RAW_H, target_w=RAW_W)

        # 2)  objects straight from the underlying OCAtari env
        oc_objects = env.envs[0].objects
        bboxes     = bboxes_from_ocatari(oc_objects)
        # ---------------------------------------------------------------------

        if bboxes:
            overlap       = measure_saliency_in_bboxes(sal_210x160, bboxes, radius=radius,
                                                       threshold_ratio=THRESHOLD_RATIO)
            total_overlap += overlap
            step_count    += 1
        else:
            overlap = 0.0

        # 3) visualisation
        if do_render:
            frame = env.render(mode="rgb_array")      # raw 210×160×3
            colour_inner = (  0,255,  0)   # green  = exact object
            colour_outer = (255,  0,  0)   # red    = object ± radius

            # pad the OCAtari boxes once
            pad_boxes = [{
                "x1": max(0, b["x1"] - radius),
                "y1": max(0, b["y1"] - radius),
                "x2": min(RAW_W-1, b["x2"] + radius),
                "y2": min(RAW_H-1, b["y2"] + radius)
            } for b in bboxes]
            left = draw_bboxes_on_image(frame, pad_boxes, color=colour_outer, thickness=2)
            left = draw_bboxes_on_image(left,  bboxes,    color=colour_inner, thickness=2)
            im_left.set_data(left)
            sal_top = keep_topk(sal_210x160, THRESHOLD_RATIO)  # keep only top 30%
            im_right.set_data(sal_top / (sal_top.max() + 1e-9))
            im_right.set_clim(0, 1)
            ax[0].set_title(f"step={step}, #Objects={len(bboxes)}")
            ax[1].set_title(f"Overlap={overlap:.2f}")
            plt.pause(0.01)

        if dones[0]:
            obs = env.reset()

    avg = total_overlap / step_count if step_count else 0.0
    print(f"Average fraction of top-salient pixels in OCAtari objects = {avg:.3f}")

def compute_saliency_for_rgb_atari(model, obs_frame_4x84x84):
    """
obs_frame_4x84x84 : (4,84,84) uint8 or float32 [0,255]
    """
    frame_tensor = torch.as_tensor(obs_frame_4x84x84, dtype=torch.float32,
                                   device=model.device).unsqueeze(0)
    frame_tensor.requires_grad_()

    q_vals = model.net(frame_tensor)          # ← direct forward
    best = q_vals.argmax(1)
    q_vals[0, best].backward()

    grad = frame_tensor.grad.abs().squeeze(0) # (4,84,84)
    sal = grad.max(0)[0].cpu().numpy()        # (84,84)
    return sal

def load_dqn_general():
    model = DQN.load(
        MODEL_PATH,
        custom_objects={
            # Fix deserialization warnings
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
            # Directly set optimize_memory_usage to False
            "optimize_memory_usage": False
        }
    )
    return model

def load_dqn(env, path):
    action_n = env.action_space.n
    net = load_atarinet(path, action_n, device="cuda:0" if torch.cuda.is_available() else "cpu")
    model = AtariDQNWrapper(net)
    return model

def overlap_with_padded_boxes(arr2d: np.ndarray,
                              bboxes: list,
                              radius: int = 15,
                              threshold_ratio: float = 0.30):
    if arr2d.size == 0 or not bboxes:
        return 0.0

    H, W = arr2d.shape
    obj_mask = np.zeros_like(arr2d, dtype=np.uint8)
    for b in bboxes:
        x1 = max(0, b["x1"] - radius)
        y1 = max(0, b["y1"] - radius)
        x2 = min(W - 1, b["x2"] + radius)
        y2 = min(H - 1, b["y2"] + radius)
        obj_mask[y1:y2 + 1, x1:x2 + 1] = 1

    k = max(1, int(arr2d.size * threshold_ratio))
    thresh = np.partition(arr2d.ravel(), -k)[-k]
    sal_mask = topk_mask(arr2d, threshold_ratio)


    return (obj_mask & sal_mask).sum() / sal_mask.sum()

def overlap_masks(mask_a: np.ndarray, mask_b: np.ndarray):
    """Jaccard index of two bool masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union else 0.0

def topk_mask(arr2d: np.ndarray, ratio: float):
    flat = arr2d.ravel()
    k = max(1, int(flat.size * ratio))
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    return mask.reshape(arr2d.shape)

def frac_inside(arr2d, bboxes, radius, ratio):
    if not bboxes: return 0.0
    H, W = arr2d.shape
    pad = [{"x1": max(0,b["x1"]-radius), "y1": max(0,b["y1"]-radius),
            "x2": min(W-1,b["x2"]+radius), "y2": min(H-1,b["y2"]+radius)}
           for b in bboxes]
    box_mask = np.zeros_like(arr2d, dtype=bool)
    for b in pad:
        box_mask[b["y1"]:b["y2"]+1, b["x1"]:b["x2"]+1] = True
    sal_mask = topk_mask(arr2d, ratio)
    return np.logical_and(box_mask, sal_mask).sum() / sal_mask.sum()


# Full evaluation loop: agent saliency vs. human gaze vs. objects
# usable but only limited so not integrated right now
def evaluate_saliency_human_object(model_agent,
                                   env,
                                   num_steps: int = 1_000,
                                   radius: int = 15,
                                   sal_ratio: float = 0.30,
                                   human_ratio: float = 0.30,
                                   do_render: bool = True):
    """
Prints three episode-level metrics:
• agent-object overlap   (fraction ∈ [0,1])
• human-object overlap   (fraction ∈ [0,1])
• mis-alignment = 1 − overlap(agent mask, human mask)
Bounding boxes are the ones currently produced by OCAtari; each is
padded by `radius` pixels before testing containment.
    """
    # ---------- live window ------------------------------------------------
    if do_render:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ims = [
            ax[0].imshow(np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8)),
            ax[1].imshow(np.zeros((RAW_H, RAW_W)), cmap="magma", vmin=0, vmax=1),
            ax[2].imshow(np.zeros((RAW_H, RAW_W)), cmap="magma", vmin=0, vmax=1),
        ]
        for a, t in zip(ax,
                        ["Frame",
                         f"Agent saliency (top {int(sal_ratio*100)}%)",
                         f"Pred. human gaze (top {int(human_ratio*100)}%)"]):
            a.set_title(t); a.axis("off")
        plt.tight_layout()

    # ---------- accumulators ----------------------------------------------
    agent_obj_sum = 0.0
    human_obj_sum = 0.0
    misalign_sum  = 0.0
    frames = 0

    obs = env.reset()
    while frames < num_steps:
        # 1) agent acts
        action, _ = model_agent.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        # 2) agent saliency & human heat-map  (84×84 → 210×160)
        sal210  = upsample_saliency(
            compute_saliency_for_rgb_dqn_net(model_agent, obs[0]),
            RAW_H, RAW_W)

        # 3) boxes from OCAtari
        bboxes = bboxes_from_ocatari(env.envs[0].objects)

        # 4) overlaps with boxes
        agent_obj_sum += overlap_with_padded_boxes(
            sal210,  bboxes, radius, sal_ratio)

        # 5) mis-alignment agent ↔ human
        mask_a = topk_mask(sal210,  sal_ratio)
        misalign_sum += 1.0# - overlap commented out because of error message

        # 6) live view
        if do_render:
            frame_rgb = env.render(mode="rgb_array")
            ims[0].set_data(frame_rgb)
            ims[1].set_data(sal210  / (sal210 .max()  + 1e-9))
            plt.pause(0.001)

        frames += 1
        if done[0]:
            obs = env.reset()

    # ---------- results ----------------------------------------------------
    print(f"\n==  Results over {frames} frames  ==")
    print(f"Agent saliency ⟂ objects (padded r={radius}) : "
          f"{agent_obj_sum / frames:.3f}")
    print(f"Human gaze    ⟂ objects (padded r={radius}) : "
          f"{human_obj_sum / frames:.3f}")
    print(f"Mis-alignment agent vs human                : "
          f"{misalign_sum  / frames:.3f}")






if __name__ == "__main__":
    # Create environment
    #model = PPO.load(MODEL_PATH)
    #model = load_dqn_general()
    #model = load_dqn(env, MODEL_PATH)

    # Evaluate
    #evaluate_attention_draw_box(model, env, num_steps=NUM_STEPS, do_render=True)


    env_scoring = make_env_for_ckpt(GAME)   # raw ALE, no OCAtari
    model      = load_dqn(env_scoring, MODEL_PATH)
    R = 0
    obs, done = env_scoring.reset(), False
    while not done:
        action, _ = model.predict(obs)        # argmax w/out saliency stuff
        obs, r, done, _ = env_scoring.step(int(action))
        R += r
    print("Episode return:", R)

    env_attn = create_env()          # OCAtari clone
    model    = load_dqn(env_attn, MODEL_PATH)  # AtariDQNWrapper
    #print("scoring:", obs.dtype, obs.max(),
     #     "attn:",    env_attn.reset().dtype,    env_attn.reset().max())

    env   = create_env(ENV_ID)                # OCAtari env, matches training
    agent = load_dqn(env, MODEL_PATH)         # AtariDQNWrapper around Atarinet
    #gaze  = HumanGaze(GAZE_WT, MEAN_NPZ)     # <-- new




    evaluate_attention_dqn_ocatari(
        model,
        env_attn,
        radius=15,
        num_steps=1000,
        do_render=True          # overlays + OCAtari object boxes
    )

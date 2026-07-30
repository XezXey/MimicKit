import sys
import os

# Ensure mimickit subfolder is in Python's import path
sys.path.insert(0, os.path.abspath("mimickit"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import anim.motion_lib as motion_lib
import anim.mjcf_char_model as mjcf_char_model

# 1. Config paths
char_file = "data/assets/smpl/smpl.xml"          # Asset file
# motion_file = "data/motions/smpl/smpl_walk.pkl"   # Path to your converted .pkl
motion_file = "./test.pkl"   # Path to your converted .pkl
output_gif = "output_motion.gif"

# 2. Load character model & motion library
device = "cpu"
char_model = mjcf_char_model.MJCFCharModel(device=device)
char_model.load(char_file)

m_lib = motion_lib.MotionLib(
    motion_file=motion_file,
    kin_char_model=char_model,
    device=device
)

# 3. Compute 3D joint/body positions across frames
print(m_lib.get_total_length(), m_lib.get_motion_lengths())
print(m_lib.get_num_motions(), m_lib.get_num_joints())
print(m_lib._motion_fps, m_lib._motion_num_frames)
num_frames = m_lib._motion_num_frames
fps = m_lib._motion_fps
# parent_indices = char_model.get_parent_indices().cpu().numpy()
parent_indices = char_model._parent_indices
print(m_lib._kin_char_model)
print(parent_indices)

all_body_pos = []
for frame_idx in range(num_frames):
    time = torch.tensor([frame_idx / fps])
    root_pos, root_rot, _, _, joint_rot, _ = m_lib.calc_motion_frame(torch.tensor([0]), time)
    if torch.isnan(root_pos).any() or torch.isnan(root_rot).any() or torch.isnan(joint_rot).any():
        print(f"NaN detected at frame {frame_idx}")
        continue
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    all_body_pos.append(body_pos[0].cpu().numpy())

all_body_pos = np.array(all_body_pos)  # Shape: (num_frames, num_joints, 3)

# 4. Render 3D animated GIF
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    pts = all_body_pos[frame]

    # Draw joint points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=25)

    # Draw bones connecting parent & child joints
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            p1 = pts[parent_idx]
            p2 = pts[child_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='blue', lw=2)

    # Set camera bounds
    center = pts[0]
    ax.set_xlim([center[0] - 1.0, center[0] + 1.0])
    ax.set_ylim([center[1] - 1.0, center[1] + 1.0])
    ax.set_zlim([0, 2.0])
    ax.set_title(f"Frame {frame}/{num_frames}")

anim = FuncAnimation(fig, update, frames=range(0, num_frames, 2), interval=1000/fps)
anim.save(output_gif, writer=PillowWriter(fps=fps/2))
print(f"Animation saved to {output_gif}")
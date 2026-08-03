import sys
import os

# Ensure mimickit subfolder is in Python's import path
sys.path.insert(0, os.path.abspath("mimickit"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle

import anim.motion_lib as motion_lib
import anim.mjcf_char_model as mjcf_char_model

# 1. Config paths
char_file = "data/assets/smpl/smpl.xml"          # Asset file
# motion_file = "data/motions/smpl/smpl_walk.pkl"   # Path to your converted .pkl
# motion_file = "data/motions/humanoid/humanoid_spinkick.pkl"   # Path to your converted .pkl
motion_file = "./soccer_x_test.pkl"   # Path to your converted .pkl
with open(motion_file, 'rb') as f:
    motion_data = pickle.load(f)
    print("Loaded motion data keys:", motion_data.keys())

for key in motion_data.keys():
    print(f"{key}: {type(motion_data[key])}, shape: {np.array(motion_data[key]).shape}")
print(motion_data['frames'][0])
print(motion_data['loop_mode'])
print(motion_data['fps'])
exit()

output_gif = "soccer_x_motion.gif"
# motion_file = "./test.pkl"   # Path to your converted .pkl
# output_gif = "output_motion.gif"

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

# 4. Render 3D animated GIF (2x2 Multi-View Grid: Front, Side, Top, 3D Perspective)
fig = plt.figure(figsize=(12, 10))
ax_front = fig.add_subplot(221, projection='3d')
ax_side  = fig.add_subplot(222, projection='3d')
ax_top   = fig.add_subplot(223, projection='3d')
ax_3d    = fig.add_subplot(224, projection='3d')

views = [
    (ax_front, "Front View", 0, -90),
    (ax_side,  "Right Side View", 0, 0),
    (ax_top,   "Top-Down View", 89, -90),
    (ax_3d,    "3D Perspective View", 20, 45)
]

actual_num_frames = len(all_body_pos)

def update(frame):
    pts = all_body_pos[frame]
    center = pts[0]  # Root center position for camera tracking
    
    for ax, title, elev, azim in views:
        ax.clear()
        
        # Draw joint points
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=18)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c='red', s=18)
        
        # Draw bones connecting parent & child joints
        for child_idx, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0:
                p1 = pts[parent_idx]
                p2 = pts[child_idx]
                # ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='blue', lw=2)
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], c='blue', lw=2)
                
        # Apply camera angles and title
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Set bounds tracking root position
        ax.set_xlim([center[0] - 1.2, center[0] + 1.2])
        ax.set_ylim([center[1] - 1.2, center[1] + 1.2])
        ax.set_zlim([0.0, 2.2])
        
        # Hide axis labels/ticks for clean visual output
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    fig.suptitle(f"Motion Verification — Frame {frame}/{actual_num_frames}", fontsize=14, fontweight='bold')

print("Rendering multi-view GIF...")
anim = FuncAnimation(fig, update, frames=range(0, actual_num_frames, 2), interval=1000/fps)
anim.save(output_gif, writer=PillowWriter(fps=fps/2))
print(f"Multi-view animation successfully saved to {output_gif}")
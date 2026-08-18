# graph_gen.py
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import roboticstoolbox as rtb
from spatialmath import SE3
from torch_geometric.data import Data

def generate_chunk(start_idx, num_samples, out_path):
    robot = rtb.models.DH.Panda()
    
    dh_params, q_min, q_max = [], [], []
    for i, link in enumerate(robot.links):
        dh_params.append([link.a, link.alpha, link.d, link.offset])
        q_min.append(robot.qlim[0, i])
        q_max.append(robot.qlim[1, i])
    dh_tensor = torch.tensor(dh_params, dtype=torch.float32)
    qlim = np.array([q_min, q_max])
    n_joints = len(robot.links)

    data_list = []
    success, fail = 0, 0

    for _ in tqdm(range(num_samples), desc=f'Chunk {start_idx}-{start_idx+num_samples-1}'):
        while True:
            q_start = np.random.uniform(qlim[0], qlim[1])
            try:
                T_start = robot.fkine(q_start)
            except:
                fail += 1
                continue

            delta_pos = np.random.uniform(-0.5, 0.5, 3)
            delta_rpy = np.random.uniform(-3, 3, 3)
            T_target = T_start * SE3(delta_pos) * SE3.RPY(delta_rpy, order='zyx')

            try:
                result = robot.ikine_LM(T_target, q0=q_start, tol=1e-4, ilimit=2000, joint_limits=True)
            except:
                fail += 1
                continue
            if not result.success:
                fail += 1
                continue

            q_goal = result.q
            if np.any(q_goal < qlim[0]) or np.any(q_goal > qlim[1]):
                fail += 1
                continue

            try:
                T_final = robot.fkine(q_goal)
                err_pos = np.linalg.norm(T_target.t - T_final.t)
                err_rot = np.linalg.norm(T_target.rpy(order='zyx') - T_final.rpy(order='zyx'))
                if err_pos > 0.01 or err_rot > 0.05:
                    fail += 1
                    continue
            except:
                fail += 1
                continue

            dq = q_goal - q_start
            for j in range(n_joints):
                dq[j] = (dq[j] + np.pi) % (2 * np.pi) - np.pi

            # Build node features
            node_features = []
            for i in range(n_joints):
                node_features.append([q_start[i], 1.0, qlim[0, i], qlim[1, i]])
            node_features.append([0.0, 0.0, 0.0, 0.0])
            x = torch.tensor(node_features, dtype=torch.float32)

            # Edges
            src_chain = list(range(n_joints - 1)) + list(range(1, n_joints))
            dst_chain = list(range(1, n_joints)) + list(range(n_joints - 1))
            virtual_idx = n_joints
            src_target, dst_target = [], []
            for j in range(n_joints):
                src_target.extend([j, virtual_idx])
                dst_target.extend([virtual_idx, j])
            edge_index = torch.tensor([src_chain + src_target, dst_chain + dst_target], dtype=torch.long)

            # Edge attributes
            edge_attr_list = []
            for i in range(n_joints - 1):
                chain_attr = torch.cat([dh_tensor[i + 1], torch.zeros(1)])
                edge_attr_list.extend([chain_attr, chain_attr.clone()])
            target_attr = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            for _ in range(2 * n_joints):
                edge_attr_list.append(target_attr.clone())
            edge_attr = torch.stack(edge_attr_list, dim=0)

            # Global feature
            target_pose_vec = np.hstack([T_target.t, T_target.rpy(order='zyx')])
            global_feat = torch.tensor(target_pose_vec, dtype=torch.float32).unsqueeze(0)

            y = torch.tensor(dq, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        global_feat=global_feat, y=y)
            data_list.append(data)
            success += 1
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, out_path)
    print(f"Chunk saved to {out_path}: {success} success, {fail} failures")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, required=True)   # путь к файлу .pt
    args = parser.parse_args()
    generate_chunk(args.start, args.count, args.out_dir)
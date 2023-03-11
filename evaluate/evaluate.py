import yaml
from tqdm import tqdm
import torch
import sys
import math
import numpy as np
import os
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils import utils
from modules.overlapnetvlad import vlad_head, overlap_head


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_descriptors(vlad, fea_folder, batch_num):
    fea_files = sorted(os.listdir(fea_folder))
    fea_files = [os.path.join(fea_folder, v) for v in fea_files]
    length = len(fea_files)

    vlad_arr = np.zeros((length, 1024), dtype=np.float32)

    for q_index in tqdm(range(length // batch_num),
                        total=length // batch_num):
        batch_files = fea_files[q_index * batch_num:(q_index + 1) * batch_num]
        queries = utils.load_npy_files(batch_files)
        input = torch.tensor(queries).float().to(device)
        vlad_out = vlad(input)
        ran = range(q_index * batch_num, (q_index + 1) * batch_num)
        vlad_arr[ran] = vlad_out.detach().cpu().numpy()

    index_edge = length // batch_num * batch_num
    if index_edge < length:
        batch_files = fea_files[index_edge:length]
        queries = utils.load_npy_files(batch_files)
        input = torch.tensor(queries).float().to(device)
        vlad_out = vlad(input)
        ran = range(index_edge, length)
        vlad_arr[ran] = vlad_out.detach().cpu().numpy()

    return vlad_arr


def evaluate_vlad(vlad, topk=25):
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    seq = config["test_config"]["seq"]
    root = config["data_root"]["data_root_folder"]
    th_min = config["test_config"]["th_min"]
    th_max = config["test_config"]["th_max"]
    th_max_pre = config["test_config"]["th_max_pre"]
    skip = config["test_config"]["skip"]
    batch_num = config["test_config"]["batch_num"]

    vlad.eval()
    fea_folder = os.path.join(root, seq, "BEV_FEA")
    vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

    pose = np.genfromtxt(os.path.join(root.replace(
        'sequences', 'poses_semantic'), seq + '.txt'))[:, [3, 11]]

    pos_nums = np.zeros((topk), dtype=np.int32)
    neg_nums = np.zeros((topk), dtype=np.int32)
    recalls = np.zeros((topk), dtype=np.float32)
    for i in tqdm(range(len(pose)), total=len(pose)):
        diff = pose - pose[i]
        pos_dis = np.linalg.norm(diff, axis=1)

        max_d = np.inf
        pos_dis[max(i - skip, 0):] = max_d
        mask = (pos_dis < th_min)
        pos_dis[mask] = max_d

        minid_gt = np.argmin(pos_dis)
        temp_min = np.min(pos_dis)
        if temp_min < th_max:
            diff = vlad_arr - vlad_arr[i]
            vlad_dis = np.linalg.norm(diff, axis=1)

            vlad_dis[max(i - skip, 0):] = max_d
            vlad_dis[mask] = max_d

            vlad_topks = np.argsort(vlad_dis)[:topk]

            for j in range(topk):
                minid = vlad_topks[j]
                mindis = pos_dis[minid]
                if mindis < th_max_pre:
                    pos_nums[j:] += 1
                    break
                else:
                    neg_nums[j] += 1
    recalls = pos_nums / (pos_nums + neg_nums)
    vlad.train()
    return recalls


def evaluate_overlapnetvlad(vlad, overlapnetvlad, topk=25, topn=1):
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    seq = config["test_config"]["seq"]
    root = config["data_root"]["data_root_folder"]
    th_min = config["test_config"]["th_min"]
    th_max = config["test_config"]["th_max"]
    th_max_pre = config["test_config"]["th_max_pre"]
    skip = config["test_config"]["skip"]
    batch_num = config["test_config"]["batch_num"]

    vlad.eval()
    fea_folder = os.path.join(root, seq, "BEV_FEA")
    feature_files = os.listdir(fea_folder)
    feature_files.sort()
    feature_files = [os.path.join(fea_folder, v) for v in feature_files]
    vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

    pose = np.genfromtxt(os.path.join(root.replace(
        'sequences', 'poses_semantic'), seq + '.txt'))[:, [3, 11]]
    length = len(pose)

    pos_nums = np.zeros((topn), dtype=np.int32)
    neg_nums = np.zeros((topn), dtype=np.int32)
    recalls = np.zeros((topn), dtype=np.float32)
    for i in tqdm(range(length), total=length):
        diff = pose - pose[i]
        pos_dis = np.linalg.norm(diff, axis=1)

        max_d = np.inf
        pos_dis[max(i - skip, 0):] = max_d
        mask = (pos_dis < th_min)
        pos_dis[mask] = max_d

        minid_gt = np.argmin(pos_dis)
        temp_min = np.min(pos_dis)
        if temp_min < th_max:
            diff = vlad_arr - vlad_arr[i]
            vlad_dis = np.linalg.norm(diff, axis=1)

            vlad_dis[max(i - skip, 0):] = max_d
            vlad_dis[mask] = max_d

            vlad_topks = np.argsort(vlad_dis)[:topk]

            feai = utils.load_npy_files([feature_files[i]])
            feai = torch.from_numpy(feai).to(device)
            scores = np.zeros(topk, dtype='float32')
            for j in range(topk):
                feaj = utils.load_npy_files([feature_files[j]])
                feaj = torch.from_numpy(feaj).to(device)
                overlap, _ = overlapnetvlad(torch.cat([feai, feaj]).permute(0,2,3,1))
                scores[j] = overlap.detach().cpu().numpy()

            for n in range(topn):
                minid = vlad_topks[n]
                mindis = pos_dis[minid]
                if mindis < th_max_pre:
                    pos_nums[n:] += 1
                    break
                else:
                    neg_nums[n] += 1
    recalls = pos_nums / (pos_nums + neg_nums)
    vlad.train()
    return recalls


if __name__ == '__main__':
    vlad = vlad_head().to(device)
    checkpoint = torch.load(os.path.join(p, "./models/vlad.ckpt"))
    vlad.load_state_dict(checkpoint['state_dict_vlad'])
    
    # print("recall@N\n", evaluate_vlad(vlad, topk=25))

    overlap = overlap_head(32).to(device)
    checkpoint = torch.load(os.path.join(p, "./models/bevnet.ckpt"))
    overlap.load_state_dict(checkpoint['state_dict'], strict=False)

    print("recall@N\n", evaluate_overlapnetvlad(vlad, overlap, topk=25, topn=1))



from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from scipy.linalg import norm
from tqdm import tqdm
import threading


class kitti_dataset(Dataset):
    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.seqs = seqs
        self.poses = []
        for seq in seqs:
            pose = np.genfromtxt(os.path.join(root.replace(
                'sequences', 'poses_semantic'), seq + '.txt'))[:, [3, 11]]
            self.poses.append(pose)
        self.pairs = {}

        key = 0
        acc_num = 0
        for i in range(len(self.poses)):
            pose = self.poses[i]
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose**2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            id_pos = np.argwhere((dis < pos_threshold) & (dis > 0))
            id_neg = np.argwhere(dis < neg_threshold)
            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(
                        negatives.tolist())}
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))
        self.traing_latent_vectors = [None] * len(self.pairs)

    def get_random_positive(self, idx):
        positives = self.pairs[idx]["positives"]
        randid = random.randint(0, len(positives) - 1)

        return positives[randid]

    def get_random_negative(self, idx):
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = random.randint(0, len(negatives) - 1)

        return negatives[randid]

    def get_random_hard_positive(self, idx):
        random_pos = self.pairs[idx]["positives"]
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_pos) - 1)
            return random_pos[randid]

        latent_vecs = []
        for j in range(len(random_pos)):
            latent_vecs.append(self.traing_latent_vectors[random_pos[j]])
 
        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        maxid = np.argmax(diff)

        return random_pos[maxid]

    def get_random_hard_negative(self, idx):
        random_neg = list(self.all_ids - self.pairs[idx]["negatives"])
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_neg) - 1)
            return random_neg[randid]

        latent_vecs = []
        for j in range(len(random_neg)):
            latent_vecs.append(self.traing_latent_vectors[random_neg[j]])

        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        minid = np.argmin(diff)

        return random_neg[minid]

    def get_other_neg(self, id_pos, id_neg):
        random_neg = list(
            self.all_ids -
            self.pairs[id_pos]["negatives"] -
            self.pairs[id_neg]["negatives"])
        randid = random.randint(0, len(random_neg) - 1)

        return random_neg[randid]

    def update_latent_vectors(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]

    def load_fea(self, idx):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "BEV_FEA", id + '.npy')

        return np.load(file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        queryid = idx % len(self.pairs)
        negid = self.get_random_hard_negative(queryid)
        posid = self.get_random_hard_positive(queryid)
        otherid = self.get_other_neg(queryid, negid)

        query_voxel = self.load_fea(queryid)
        pos_voxel = self.load_fea(posid)
        neg_voxel = self.load_fea(negid)
        other_voxel = self.load_fea(otherid)

        return {
            "id": queryid,
            "query_desc": query_voxel,
            "pos_desc": pos_voxel,
            "neg_desc": neg_voxel,
            "other_desc": other_voxel}


if __name__ == "__main__":
    pass

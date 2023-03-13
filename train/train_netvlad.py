from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
import yaml
import time
import os
import random
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from tools.database import kitti_dataset
from modules.loss import quadruplet_loss
from modules.overlapnetvlad import vlad_head
from evaluate import evaluate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
randg = np.random.RandomState()


def train(config):
    root = config["data_root"]["data_root_folder"]
    log_folder = config["training_config"]["log_folder"]
    training_seqs = config["training_config"]["training_seqs"]
    pretrained_vlad_model = config["training_config"]["pretrained_vlad_model"]
    pos_threshold = config["training_config"]["pos_threshold"]
    neg_threshold = config["training_config"]["neg_threshold"]
    batch_size = config["training_config"]["batch_size"]
    epoch = config["training_config"]["epoch"]
    
    log_folder = os.path.join(p, log_folder)
    if (not os.path.exists(log_folder)):
        os.makedirs(log_folder)

    writer = SummaryWriter()
    train_dataset = kitti_dataset(
        root=root,
        seqs=training_seqs,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=1)
    vlad = vlad_head().to(device=device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vlad.parameters()),
        lr=1e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[200000, 3200000, 51200000, 1638400000],
        gamma=0.1)
    loss_function = quadruplet_loss

    
    if not pretrained_vlad_model == "":
        checkpoint = torch.load(pretrained_vlad_model)
        vlad.load_state_dict(checkpoint['state_dict_vlad'])

    batch_num = 0
    for i in range(epoch):
        vlad.train()
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(
                train_loader), desc='Train epoch ' + str(i), leave=False):
            optimizer.zero_grad()
            input = torch.cat([sample_batch['query_desc'],
                               sample_batch['pos_desc'],
                               sample_batch['neg_desc'],
                               sample_batch['other_desc']], dim=0).to(device)
            out = vlad(input)

            query_fea, pos_fea, neg_fea, other_fea = torch.split(
                out, [batch_size, batch_size, batch_size, batch_size], dim=0)
            train_dataset.update_latent_vectors(
                query_fea.detach().cpu().numpy(),
                sample_batch['id'].detach().cpu().numpy())

            pos_dis, neg_dis, other_dis, loss = loss_function(
                query_fea, pos_fea, neg_fea, other_fea, 0.5, 0.2)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar(
                    'loss', loss.cpu().item(), global_step=batch_num)
                writer.add_scalar(
                    'LR',
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    global_step=batch_num)
                writer.add_scalar('pos_dis',
                                  pos_dis.cpu()[0].item(),
                                  global_step=batch_num)
                writer.add_scalar('neg_dis',
                                  neg_dis.cpu()[0].item(),
                                  global_step=batch_num)
                writer.add_scalar('other_dis',
                                  other_dis.cpu()[0].item(),
                                  global_step=batch_num)

                batch_num += 1

        recall = evaluate.evaluate_vlad(vlad)
        print("EVAL RECALL:", recall)
        writer.add_scalar("RECALL",
                          recall,
                          global_step=batch_num)
        torch.save({'epoch': i,
                    'state_dict_vlad': vlad.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'batch_num': batch_num},
                   os.path.join(log_folder, str(recall) + ".ckpt"))
        scheduler.step()


if __name__ == '__main__':
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    train(config)

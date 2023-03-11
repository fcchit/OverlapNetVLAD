import torch


def quadruplet_loss(q_vec, pos_vec, neg_vec, other_neg, m1, m2):
    pos_dis = ((q_vec - pos_vec)**2).sum(dim=1)
    neg_dis = ((q_vec - neg_vec)**2).sum(dim=1)
    other_dis = ((neg_vec - other_neg)**2).sum(dim=1)
    triplet_loss = m1 + pos_dis - neg_dis
    triplet_loss = triplet_loss.clamp(min=0.0)
    second_loss = m2 + pos_dis - other_dis
    second_loss = second_loss.clamp(min=0.0)
    sum_loss = triplet_loss + second_loss
    mask = (sum_loss > 0)
    return pos_dis, neg_dis, other_dis, torch.sum(
        sum_loss) / (torch.sum(mask) + 1e-6)
    # return torch.mean(triplet_loss + second_loss)


if __name__ == "__main__":
    pass

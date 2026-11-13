import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(real_outputs, gen_outputs):
    loss = 0
    for dr, dg in zip(real_outputs, gen_outputs):
        loss = loss + torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
    return loss


def generator_loss(gen_outputs):
    return sum(torch.mean((1 - dg.float()) ** 2) for dg in gen_outputs)

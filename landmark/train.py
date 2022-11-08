import os
import torch
from torch import optim
import numpy as np

from utils.loss import euclidean_losses, js_reg_losses, average_loss
from utils.dataloads import getDataloader
from model.coordnet import CoordRegressionNetwork
from log import Log


def train():
    net = CoordRegressionNetwork(n_locations=1,
                                 net="unet",
                                 conv_depths=(8, 16, 32, 64),
                                 midch=8).to(device)
    optimizer = optim.Adam(net.parameters(),
                           lr=LR,
                           betas=(0.9, 0.999),
                           weight_decay=1e-5)
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=LR,
    #                       momentum=0.9,
    #                       weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                            T_0=5,
    #                                                            T_mult=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    minloss = np.inf
    train_dataloader, test_dataloader = getDataloader(root_dir)

    for epoch in range(EPOCH):
        batches_done = 1
        loss_sum = 0

        for i, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)
            coords, heatmaps = net(data)

            # cal loss
            euc_losses = euclidean_losses(coords, label)
            reg_losses = js_reg_losses(heatmaps, label, sigma_t=1.0)
            loss = average_loss(euc_losses + reg_losses)
            loss_sum += loss

            log.info(
                f"[LR: {scheduler.get_last_lr()[0]:.6f}] [EPOCH {epoch + 1}/{EPOCH}] [BATCH {batches_done} / {len(train_dataloader)}] [Loss: {loss:.6f}]"
            )

            # Calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batches_done % 10 == 0:
                print(coords.detach().cpu().numpy())
                print(label.cpu().numpy())
                print('=' * 50)
            batches_done += 1

        scheduler.step()
        log.info(
            f"[EPOCH {epoch + 1}/{EPOCH}] [training loss: {loss_sum:.6f}]")

        test_loss = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_dataloader):
                data, label = data.to(device), label.to(device)
                coords, heatmaps = net(data)

                euc_losses = euclidean_losses(coords, label)
                reg_losses = js_reg_losses(heatmaps, label, sigma_t=1.0)
                loss = average_loss(euc_losses + reg_losses)
                test_loss += loss

                log.info(
                    f"[EPOCH {epoch + 1}/{EPOCH}] [BATCH {i+1} / {len(test_dataloader)}] [Loss: {loss:.6f}]"
                )
        log.info(
            f"[EPOCH {epoch + 1}/{EPOCH}] [testing loss: {test_loss:.6f}]")

        infos = {}
        infos["epoch"] = epoch
        infos["scheduler"] = scheduler
        infos["net"] = net.state_dict()
        infos["optim"] = optimizer
        if test_loss < minloss:
            minloss = test_loss
            best_pth_path = os.path.join(result_path, "best_coords.pth")
            torch.save(infos, best_pth_path)
            log.info(
                f'$$$$ Save Epoch: [{epoch+1}] [{best_pth_path}] Success $$$$')

        if (epoch + 1) % 50 == 0:
            save_epoch = os.path.join(result_path,
                                      f"best_coords_{epoch + 1}.pth")
            torch.save(infos, save_epoch)
            log.info(f'--------------Save [{save_epoch}] Success-------------')


if __name__ == "__main__":
    log = Log(filename="landmark/log/res.log", mode="w").getlog()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    result_path = "landmark/result"
    os.makedirs(result_path, exist_ok=True)

    root_dir = "/home/wangs/traindata" if os.path.exists(
        "/home") else "Y:/traindata"

    EPOCH = 100
    LR = 1e-2
    train()
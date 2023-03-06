import os
import torch
from torch import nn
from torch.optim import lr_scheduler, Adam, SGD
import numpy as np
from log import Log
from utils import losses, dataloads
from models import net
from config import Config as conf


def train(model, recon, epoch, train_iter, loss_dict, scheduler, optimizer):
    batches_done = 1
    train_loss = 0
    model.train()
    for i, (imgA, imgB, filedir) in enumerate(train_iter):
        imgA, imgB, = imgA.to(device), imgB.to(device)
        rt = model(imgA, imgB)
        warped = recon(imgB, rt)

        loss_BA = losses.pearson_correlation(imgA, warped)

        loss = loss_BA

        train_loss += loss.item()

        loss_dict['train_loss'].append(np.round(loss.item(), 4))
        log.info(
            f"[filedir: {filedir}][LR: {scheduler.get_last_lr()[0]:.6f}] [EPOCH {epoch + 1}/{conf.epochs}] [BATCH {batches_done} / {len(train_iter)}] [Loss: {loss:.4f}]"
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batches_done += 1
    return train_loss


def validation(model, recon, epoch, valid_iter, loss_dict):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (imgA, imgB, filedir) in enumerate(valid_iter):
            imgA, imgB, = imgA.to(device), imgB.to(device)
            rt = model(imgA, imgB)
            warped = recon(imgB, rt)

            loss_BA = losses.pearson_correlation(imgA, warped)

            loss = loss_BA
            valid_loss += loss.item()

            log.info(
                f"[filedir: {filedir}][EPOCH {epoch + 1}/{conf.epochs}] [BATCH {i+1} / {len(valid_iter)}] [Loss: {loss:.4f}]"
            )

            loss_dict['valid_loss'].append(np.round(loss.item(), 4))
    return valid_loss


def test(model, recon, epoch, test_iter, loss_dict, scheduler, optimizer):
    test_loss = 0
    # model.eval()
    # with torch.no_grad():
    for i, (imgA, imgB, filedir) in enumerate(test_iter):
        imgA, imgB, = imgA.to(device), imgB.to(device)
        rt = model(imgA, imgB)
        warped = recon(imgB, rt)

        loss_BA = losses.pearson_correlation(imgA, warped)

        loss = loss_BA
        test_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.info(
            f"[filedir: {filedir}][EPOCH {epoch + 1}/{conf.epochs}] [BATCH {i+1} / {len(test_iter)}] [Loss: {loss:.4f}]"
        )

        loss_dict['test_loss'].append(np.round(loss.item(), 4))
    return test_loss


def main(loadpth=None):
    start_epoch = 0
    minloss = np.inf
    loss_dict = {'train_loss': [], 'valid_loss': [], 'test_loss': []}

    model = net.Net(8, 4).to(device)
    recon = net.SpatialTransform(mode="bilinear").to(device)

    log.info(conf.getinfo())
    dic = torch.load(loadpth)
    model.load_state_dict(dic["net"])
    for name, w in model.named_parameters():
        if "fc" in name:
            if 'weight' in name:
                nn.init.normal_(w, std=0.0001)
            elif "bias" in name:
                nn.init.zeros_(w)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=conf.lr,
                                 weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=conf.step_size,
                                    gamma=conf.gamma)

    for epoch in range(start_epoch, conf.epochs):

        # train
        # train_loss = train(model, recon, epoch, train_iter, loss_dict,
        #                    scheduler, optimizer)
        # log.info(
        #     f"[EPOCH {epoch + 1}/{conf.epochs}] [train Loss: {train_loss:.4f}]"
        # )
        # # valid
        # valid_loss = validation(model, recon, epoch, valid_iter, loss_dict)
        # log.info(
        #     f"[EPOCH {epoch + 1}/{conf.epochs}] [valid Loss: {valid_loss:.4f}]"
        # )

        # test
        test_loss = test(model, recon, epoch, test_iter, loss_dict, scheduler, optimizer)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [test Loss: {test_loss:.4f}]")

        # update lr
        scheduler.step()

        infos = {}
        infos["epoch"] = epoch + 1
        infos["scheduler"] = scheduler
        infos[f"net"] = model.state_dict()
        infos["optim"] = optimizer
        if test_loss < minloss:
            minloss = test_loss
            infos["loss"] = minloss
            torch.save(infos, bestpth_name)
            log.info(
                f'$$$$ Save Epoch: [{epoch+1}] [{bestpth_name}] Success $$$$')

        infos["loss"] = minloss
        if (epoch + 1) % conf.save_epoch == 0:
            save_epoch = os.path.join(result_path,
                                      f"{conf.save_name}_{epoch + 1}.pth")
            torch.save(infos, save_epoch)
            log.info(f'-------- Save [{save_epoch}] Success --------')


if __name__ == "__main__":

    base_path = os.path.join(os.getcwd(), conf.save_name)
    result_path = os.path.join(base_path, "results")
    os.makedirs(result_path, exist_ok=True)

    log = Log(filename=os.path.join(result_path, "train.log"),
              mode="w").getlog()
    bestpth_name = os.path.join(result_path, f"best_{conf.save_name}.pth")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # dataloads_pre
    log.info("preprocessing datas...")
    # root_dir_train = os.path.join(conf.root_path, "traindata2")
    # train_iter, valid_iter = dataloads.getDataloader(root_dir_train)
    root_dir_test = os.path.join(conf.root_path, "testdata")
    test_iter = dataloads.getTestloader(root_dir_test)
    log.info("load datas done.")

    loadpth = "/home/wangs/Exp-New/part6_3_seg/result/la2.5/best_part6_3_seg_B2A.pth"
    main(loadpth)

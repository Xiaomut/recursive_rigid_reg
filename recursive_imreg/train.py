import os
import torch
from torch import nn
from torch.optim import lr_scheduler, Adam, SGD, RMSprop
import numpy as np
from log import Log
from utils import losses, dataloads, dataloads_pre
from models import recurnet_4img, recurnet_cbct
from config import Config as conf


def train(model, epoch, train_iter, loss_dict, scheduler, optimizer):
    batches_done = 1
    train_loss = 0
    model.train()
    for i, (imgA, imgB, imgA_gt, imgB_gt, filedir) in enumerate(train_iter):
        imgA, imgB, = imgA.to(device), imgB.to(device)
        imgA_gt, imgB_gt = imgA_gt.to(device), imgB_gt.to(device)
        warped, flows, warped_gt = model(imgA, imgB, imgA_gt, imgB_gt)
        # if i % 5 == 0:
        #     print(flows)
        loss_BA = losses.pearson_correlation(imgA * imgA_gt,
                                             warped[-1] * warped_gt[-1])
        # loss_BA = losses.pearson_correlation(imgA, warped[-1])

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


def validation(model, epoch, valid_iter, loss_dict):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (imgA, imgB, imgA_gt, imgB_gt,
                filedir) in enumerate(valid_iter):
            imgA, imgB, = imgA.to(device), imgB.to(device)
            imgA_gt, imgB_gt = imgA_gt.to(device), imgB_gt.to(device)
            warped, flows, warped_gt = model(imgA, imgB, imgA_gt, imgB_gt)

            loss_BA = losses.pearson_correlation(imgA * imgA_gt,
                                                 warped[-1] * warped_gt[-1])
            # loss_BA = losses.pearson_correlation(imgA, warped[-1])

            loss = loss_BA
            valid_loss += loss.item()

            log.info(
                f"[filedir: {filedir}][EPOCH {epoch + 1}/{conf.epochs}] [BATCH {i+1} / {len(valid_iter)}] [Loss: {loss:.4f}]"
            )

            loss_dict['valid_loss'].append(np.round(loss.item(), 4))
    return valid_loss


def test(model, epoch, test_iter, loss_dict):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (imgA, imgB, imgA_gt, imgB_gt, filedir) in enumerate(test_iter):
            imgA, imgB, = imgA.to(device), imgB.to(device)
            imgA_gt, imgB_gt = imgA_gt.to(device), imgB_gt.to(device)
            warped, flows, warped_gt = model(imgA, imgB, imgA_gt, imgB_gt)

            loss_BA = losses.pearson_correlation(imgA * imgA_gt,
                                                 warped[-1] * warped_gt[-1])
            # loss_BA = losses.pearson_correlation(imgA, warped[-1])

            loss = loss_BA
            test_loss += loss.item()

            log.info(
                f"[filedir: {filedir}][EPOCH {epoch + 1}/{conf.epochs}] [BATCH {i+1} / {len(test_iter)}] [Loss: {loss:.4f}]"
            )

            loss_dict['test_loss'].append(np.round(loss.item(), 4))
    return test_loss


def main(loadpth=None):
    start_epoch = 0
    minloss = np.inf
    loss_dict = {'train_loss': [], 'valid_loss': [], 'test_loss': []}

    # model = recurnet_4img.RecursiveCascadeNetwork(device=device,
    #                                               midch=conf.channel,
    #                                               n_cascades=conf.n_cascades)
    model = recurnet_cbct.RecursiveCascadeNetwork(device=device,
                                                  midch1=conf.channel,
                                                  midch2=conf.channel2,
                                                  normalize_features=True,
                                                  normalize_matches=True,
                                                  n_cascades=conf.n_cascades)

    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    if loadpth is None:
        log.info(conf.getinfo())
        optimizer = Adam(trainable_params, lr=conf.lr, betas=(0.5, 0.999))  #
        scheduler = lr_scheduler.StepLR(optimizer, conf.step_size, conf.gamma)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, 5)
    else:
        dic = torch.load(loadpth)
        start_epoch = dic["epoch"]
        for i, submodel in enumerate(model.stems):
            submodel.load_state_dict(dic[f"cascade_{i}"])
        scheduler = dic["scheduler"]
        optimizer = dic["optim"]
        minloss = dic["loss"]

    for epoch in range(start_epoch, conf.epochs):

        # train
        train_loss = train(model, epoch, train_iter, loss_dict, scheduler,
                           optimizer)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [train Loss: {train_loss:.4f}]"
        )
        # valid
        valid_loss = validation(model, epoch, valid_iter, loss_dict)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [valid Loss: {valid_loss:.4f}]"
        )

        # test
        test_loss = validation(model, epoch, test_iter, loss_dict)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [test Loss: {test_loss:.4f}]")

        # update lr
        scheduler.step()

        infos = {}
        infos["epoch"] = epoch + 1
        infos["scheduler"] = scheduler
        for i, submodel in enumerate(model.stems):
            infos[f"cascade_{i}"] = submodel.state_dict()
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
    result_path = os.path.join(base_path, f"cas{conf.n_cascades}",
                               "cur_0832")
    os.makedirs(result_path, exist_ok=True)

    log = Log(filename=os.path.join(result_path, "train.log"),
              mode="a").getlog()
    bestpth_name = os.path.join(result_path, f"best_{conf.save_name}.pth")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # dataloads_pre
    log.info("preprocessing datas...")
    root_dir_train = os.path.join(conf.root_path, "traindata2")
    train_iter, valid_iter = dataloads.getDataloader(root_dir_train)
    root_dir_test = os.path.join(conf.root_path, "testdata")
    test_iter = dataloads.getTestloader(root_dir_test)
    log.info("load datas done.")

    loadpth = None
    main(loadpth)

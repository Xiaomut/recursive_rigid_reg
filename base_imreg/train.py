import os
import torch
from torch import nn
from torch.optim import lr_scheduler, Adam, SGD, RMSprop
import numpy as np
from log import Log
from utils import dataloads
from models.model import Net
from config import Config as conf


def train(model, epoch, criterion, train_iter, loss_dict, scheduler,
          optimizer):
    batches_done = 1
    train_loss = 0
    model.train()

    for i, (imgA, imgB, label, filedir) in enumerate(train_iter):
        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)
        fake_degree = model(imgA, imgB)
        loss = criterion(label, fake_degree)
        loss_sum += loss
        loss_dict['train_loss'].append(np.round(loss.item(), 4))
        log.info(
            f"[filedir: {filedir}][LR: {scheduler.get_last_lr()[0]:.6f}] [EPOCH {epoch + 1}/{conf.epochs}] [BATCH {batches_done} / {len(train_dataloader)}] [Loss: {loss:.6f}]"
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batches_done % 10 == 0:
        #     print(fake_degree.detach().cpu().numpy())
        #     print(label.cpu().numpy())
        #     print('=' * 50)
        batches_done += 1
    return train_loss


def validation(model, epoch, criterion, valid_iter, loss_dict):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for i, (imgA, imgB, label, filedir) in enumerate(valid_iter):
            imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(
                device)

            fake_degree = model(imgA, imgB)

            curr_loss = criterion(label, fake_degree)
            log.info(
                f"[filedir: {filedir}][EPOCH {epoch + 1}/{conf.epochs}] [BATCH {i+1} / {len(test_dataloader)}] [Loss: {curr_loss:.4f}]"
            )
            valid_loss += curr_loss

            loss_dict['valid_loss'].append(np.round(curr_loss.item(), 4))
    return valid_loss


def main():
    start_epoch = 0
    minloss = np.inf
    model = Net(mid_ch=conf.midch, growth_rate=conf.growthrate).to(device)
    # net.load_state_dict(
    #     torch.load(os.path.join(conf.result_path, f"{conf.save_name}.pth")))
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=conf.lr,
                                 betas=(conf.b1, conf.b2),
                                 weight_decay=conf.decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.mome, weight_decay=conf.decay)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=conf.step_size,
                                    gamma=conf.gamma)

    loss_dict = {'train_loss': [], 'valid_loss': []}

    for epoch in range(start_epoch, conf.epochs):
        # train
        train_loss = train(model, epoch, criterion, train_iter, loss_dict,
                           scheduler, optimizer)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [train Loss: {train_loss:.4f}]"
        )
        # valid
        valid_loss = validation(model, epoch, criterion, valid_iter, loss_dict)
        log.info(
            f"[EPOCH {epoch + 1}/{conf.epochs}] [valid Loss: {valid_loss:.4f}]"
        )

        # update lr
        scheduler.step()

        infos = {}
        infos["epoch"] = epoch + 1
        infos["scheduler"] = scheduler
        infos["optim"] = optimizer
        if valid_loss < minloss:
            minloss = valid_loss
            best_pth_path = os.path.join(result_path,
                                         f"{conf.save_name}_{mode}.pth")
            torch.save(net.state_dict(), best_pth_path)
            log.info(
                f'$$$$ Save Epoch: [{epoch+1}] [{best_pth_path}] Success $$$$')

        if (epoch + 1) % conf.save_epoch == 0:
            save_epoch = os.path.join(conf.result_path,
                                      f"{conf.save_name}_{epoch + 1}.pth")
            torch.save(net.state_dict(), save_epoch)
            log.info(f'--------------Save [{save_epoch}] Success-------------')


if __name__ == "__main__":

    mode = "B2A"  # A2B
    base_path = os.path.join(os.getcwd(), conf.save_name)
    result_path = os.path.join(base_path, "pth")
    os.makedirs(result_path, exist_ok=True)

    log = Log(filename=os.path.join(result_path, "train.log"),
              mode="w").getlog()
    bestpth_name = os.path.join(result_path, f"best_{conf.save_name}.pth")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # dataloads_pre
    log.info("loading datas...")
    root_dir_train = os.path.join(conf.root_path, "traindata")
    train_iter, valid_iter = dataloads.getDataloader(root_dir_train,
                                                     size=256,
                                                     mode=mode)
    # root_dir_test = os.path.join(conf.root_path, "testdata")
    # test_iter = dataloads.getTestloader(root_dir_test)
    log.info("load datas done.")
    main()

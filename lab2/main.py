import torch
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

from tvid import TvidDataset
from detector import *
from utils import compute_iou


lr = 5e-3
batch = 32
epochs = 60
freeze_epochs = epochs//2
device = "cuda" if torch.cuda.is_available() else "cpu"
iou_thr = 0.5


def train_epoch(detector, dataloader, criterion, optimizer, scheduler, epoch, device):
    detector.model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total, c, t = 0, 0, 0, 0
    i, count, flag = 0, 0, True
    label = list(dataloader.dataset.label)
    for p in detector.model.extractor.parameters():
        p.requires_grad = False
    detector.model.freeze_bn()
    netTrainer = trainer(detector.model, optimizer[0])
    total_loss = 0
    for X, y in bar:
        count += 1
        if flag and count >= freeze_epochs:
            flag, i = False, 1
            netTrainer.optimizer = optimizer[1]

        # TODO Implement the train pipeline.

        loss, c, t = netTrainer.train_step(X[0], X[1], y, 1.)

        # End of todo
        total_loss += loss
        correct += c
        total += t
        bar.set_postfix_str(f'lr={scheduler[i].get_last_lr()[0]:.4f} acc={correct / total * 100:.2f} loss={loss.item():.2f}')
    scheduler[i].step()
    torch.save(detector.model.state_dict(), 'Total_Loss%.4f-epoch%d.pth' % (total_loss / (epoch + 1), count))


def test_epoch(detector, dataloader, device, epoch):
    detector.model.eval()
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        for X, y in dataloader:

            # TODO Implement the test pipeline.

            pass

            # End of todo

        # print(f' val acc: {correct / total * 100:.2f}')


def main():
    trainloader = data.DataLoader(TvidDataset(root='./tiny_vid', mode='train'),
                                  batch_size=batch, shuffle=True, num_workers=4)
    testloader = data.DataLoader(TvidDataset(root='./tiny_vid', mode='test'),
                                 batch_size=batch, shuffle=True, num_workers=4)
    detector = Detector(classLabels=trainloader.dataset.label, backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512))
    optimizer = torch.optim.SGD(detector.model.parameters(), lr=lr/2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    optimizer_latter = torch.optim.SGD(detector.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler_latter = torch.optim.lr_scheduler.StepLR(optimizer_latter, 1, gamma=0.95, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_epoch(detector, trainloader, criterion, [optimizer, optimizer_latter], [scheduler, scheduler_latter], epoch, device)
        # test_epoch(detector, testloader, device, epoch)


if __name__ == '__main__':
    main()

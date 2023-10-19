import hydra
import numpy as np
import torch
from accelerate import Accelerator
from monai.metrics import CumulativeAverage
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
from dataset import WindDataset, read_and_preprocessing
from model import WindNet
from utils import get_class


def image_from_batch(batch, idx):
    img = torch.permute(batch[idx], (1, 2, 0)).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="windDL",

            # track hyperparameters and run metadata
            config=OmegaConf.to_container(cfg)
        )
        wandb.define_metric("fg_loss/*", summary="min")
        wandb.define_metric("bg_loss/*", summary="min")

    data = read_and_preprocessing(cfg)
    p_all = np.array([x for exp in data for x in exp['data'][2]])
    p_max, p_min = np.max(p_all), np.min(p_all)

    num_trains = int(len(data) * 0.8)

    train_ds = WindDataset(cfg, data[:num_trains], p_max, p_min)
    test_ds = WindDataset(cfg, data[num_trains:], p_max, p_min)

    train_dl = DataLoader(train_ds,
                          shuffle=True,
                          batch_size=cfg.general.batch_size,
                          num_workers=cfg.general.num_workers
                          )
    test_dl = DataLoader(test_ds,
                         shuffle=True,
                         batch_size=cfg.general.batch_size,
                         num_workers=cfg.general.num_workers
                         )

    # import matplotlib.pyplot as plt
    #
    # x, xs, y, y_mask = train_ds[0]
    #
    # x = torch.permute(x, (1, 2, 0)).cpu().numpy()
    # y = torch.permute(y, (1, 2, 0)).cpu().numpy()
    # y_mask = torch.permute(y_mask, (1, 2, 0)).cpu().numpy()
    #
    # plt.imshow(np.vstack((np.hstack((x, y)), np.hstack((x, y_mask)))))
    # plt.axis('off')
    # plt.show()
    #
    # exit(0)

    model = WindNet(cfg)
    optim = get_class(cfg.train.optim.type)(model.parameters(), **cfg.train.optim.params)

    model, optim, train_dl, test_dl = accelerator.prepare(model, optim, train_dl, test_dl,
                                                          device_placement=[True, True, True, True])

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optim,
        base_lr=cfg.train.optim.params.lr * 0.001,
        max_lr=cfg.train.optim.params.lr,
        step_size_up=len(train_dl) * 5,
        cycle_momentum=False
    )

    def loss_function(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        return (torch.sum(torch.abs(pred - target) * mask, [1, 2, 3]) / torch.sum(mask, [1, 2, 3]),
                torch.sum(torch.abs(pred * (1.0 - mask)), [1, 2, 3]) / torch.sum(1.0 - mask, [1, 2, 3]))

    bg_metric = CumulativeAverage()
    fg_metric = CumulativeAverage()

    progress_bar = trange(cfg.general.num_epochs * (len(train_dl) + len(test_dl)), desc="Epoch")
    for epoch in range(cfg.general.num_epochs):

        result = {'epoch': epoch}

        bg_metric.reset()
        fg_metric.reset()
        model.train()
        for x, xs, y, y_mask in train_dl:
            y_pred = model(x, xs)
            fg_loss, bg_loss = loss_function(pred=y_pred, target=y, mask=y_mask)
            accelerator.backward(fg_loss.mean() + bg_loss.mean())
            optim.step()
            scheduler.step()
            optim.zero_grad()

            bg_metric.append(torch.mean(bg_loss), x.shape[0])
            fg_metric.append(torch.mean(fg_loss), x.shape[0])

            progress_bar.update(1)

        result['bg_loss/train'] = bg_metric.aggregate()
        result['fg_loss/train'] = fg_metric.aggregate()

        bg_metric.reset()
        fg_metric.reset()
        model.eval()
        for x, xs, y, y_mask in test_dl:
            with torch.no_grad():
                y_pred = model(x, xs)
                fg_loss, bg_loss = loss_function(pred=y_pred, target=y, mask=y_mask)

                bg_metric.append(torch.mean(bg_loss), x.shape[0])
                fg_metric.append(torch.mean(fg_loss), x.shape[0])

            progress_bar.update(1)

        result['bg_loss/test'] = bg_metric.aggregate()
        result['fg_loss/test'] = fg_metric.aggregate()

        if accelerator.is_main_process:
            images = []
            for img_id, index in enumerate(np.random.permutation(len(x))):
                img = image_from_batch(x, index)
                pred = image_from_batch(y_pred, index)
                gt = image_from_batch(y, index)
                images.append(np.vstack((img, pred, gt)))
                if img_id > 0:
                    break
            result['image'] = wandb.Image(np.hstack(images))
            wandb.log(result)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()

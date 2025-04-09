import torch, time, logging
import torch.optim as optim
from utils.params import par, str_to_bool
from utils.test import test
import numpy as np
from utils.hdf5dataloader_interpolation_par5 import Dataset
from torch.utils.data import DataLoader
from utils.model_class import Par_MergeTime
import torch.nn as nn

def load_dataset(hdf5_path, batch_size, waveform_len, device, interpolation):
    dataset_train = Dataset(
        hdf5_path,
        "train",
        waveform_len,
        mode="train_wave_len",
        device=device,
        interpolation=interpolation,
    )
    dataset_test = Dataset(
        hdf5_path,
        "test",
        waveform_len,
        mode="train_wave_len",
        device=device,
        interpolation=interpolation,
    )
    if device is not None:
        num_workers = 0
    else:
        num_workers = 64
    train_ds = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_ds = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_ds, test_ds


def train(model, batch, device, optimizer):
    par, *_, wavelen = batch
    par = par.to(device).float()
    wavelen = wavelen.to(device).float()

    optimizer.zero_grad()

    x = model(par)
    loss_fn = nn.L1Loss()
    loss_total = loss_fn(x, wavelen)
    loss_total.backward()
    optimizer.step()
    return loss_total.item()


def test(model, test_ds, device, epoch):
    test_loss = 0

    for ii, batch in enumerate(test_ds):
        par, *_, wavelen = batch

        par = par.to(device).float()
        wavelen = wavelen.to(device).float()
        x = model(par)
        loss_fn = nn.L1Loss()
        test_loss += loss_fn(x, wavelen).item()
        print(
            f"Epoch [{epoch+1}]|log10 loss:{np.log10(loss_item):.4f}|MSE Loss:{loss_item:.9f}|\r",
            end="",
        )

    test_loss = test_loss / (ii + 1)

    return test_loss


if __name__ == "__main__":
    par_args = par()
    logging.basicConfig(
        filename=par_args.save_log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    device = torch.device(
        f"cuda:{par_args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    model = Par_MergeTime(
        par_args.parameter_dim, par_args.hidden_dim, par_args.hidden_layer, v2=False
    ).to(device)
    logging.info(model)

    if str_to_bool(par_args.load_check_point):
        model.load_state_dict(
            torch.load(par_args.model_checkpoint, map_location=device)["model"],
            strict=False,
        )
        logging.info("loaded checkpoint")

    optimizer = optim.Adam(model.parameters(), lr=par_args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

    train_ds, test_ds = load_dataset(
        par_args.dataset_path,
        par_args.batch_size,
        par_args.waveform_len,
        device=device,
        interpolation=str_to_bool(par_args.interpolation),
    )

    for epoch in range(par_args.num_epoch):
        train_loss = 0
        start_time = time.time()

        # train
        model.train()
        for ii, batch in enumerate(train_ds):
            loss_item = train(model, batch, device, optimizer)
            train_loss += loss_item  # model.total_loss.item()
            print(
                f"Epoch [{epoch+1}]|log10 loss:{(loss_item):.2f}|MSE Loss:{loss_item:.9f}|\r",
                end="",
            )

        # test
        with torch.no_grad():
            model.eval()
            test_loss = test(model, test_ds, device, epoch)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        scheduler.step()

        # save checkpoint
        if (epoch + 1) % 50 == 0 or epoch == par_args.num_epoch:
            torch.save({"model": model.state_dict()}, par_args.save_checkpoint_path)
            logging.info("model saved!")

        end_time = time.time()

        logging.info(
            f"Epoch [{epoch+1}/{par_args.num_epoch}]|Loss:{(train_loss/(ii+1))}|test loss:{(test_loss)}|lr:{lr}|Time:{end_time-start_time:.2f}s|"
        )

        print(
            f"Epoch [{epoch+1}/{par_args.num_epoch}]|Loss:{(train_loss/(ii+1)):.2f}|test loss:{(test_loss):.2f}|lr:{lr:.8f}|Time:{end_time-start_time:.2f}s|"
        )

    logging.info("finish training!")

import torch, time, logging
import torch.optim as optim
import torch.nn.functional as F
from utils.params import par, str_to_bool
from utils.test import test
import numpy as np
from utils.plot_test import plot_train
from utils.hdf5dataloader_interpolation_par5 import Dataset
from torch.utils.data import DataLoader
from utils.model_class import Interpolation as Interpolation

def load_dataset(hdf5_path, batch_size, waveform_len, device, interpolation):
    dataset_train = Dataset(
        hdf5_path,
        "train",
        waveform_len,
        mode="train_amp",
        device=device,
        interpolation=interpolation,
    )
    dataset_test = Dataset(
        hdf5_path,
        "test",
        waveform_len,
        mode="train_amp",
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
    model.train()
    par, label = batch
    par = par.to(device).float()
    label = label.to(device).float()

    optimizer.zero_grad()

    x = model(par)
    loss_total = F.mse_loss(x, label)
    loss_total.backward()
    optimizer.step()
    return loss_total.item()


def test(model, test_ds, device, epoch, save_image):
    test_loss = 0
    model.eval()
    for ii, batch in enumerate(test_ds):
        par, label = batch

        par = par.to(device).float()
        label = label.to(device).float()
        x = model(par)

        test_loss += F.mse_loss(x, label).item()
        print(
            f"Epoch [{epoch+1}]|log10 loss:{np.log10(test_loss):.4f}|MSE Loss:{test_loss:.9f}|\r",
            end="",
        )

    test_loss = test_loss / (ii + 1)
    plot_train(x, label, save_image)

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
    model = Interpolation(
        par_args.parameter_dim,
        par_args.hidden_dim,
        par_args.hidden_layer,
        par_args.waveform_len,
        bn=False,
    ).to(device)
    logging.info(model)

    if str_to_bool(par_args.load_check_point):
        model.load_state_dict(
            torch.load(par_args.model_checkpoint, map_location=device)["model"],
            strict=False,
        )
        logging.info("loaded checkpoint")

    optimizer = optim.Adam(model.parameters(), lr=par_args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=100,eta_min=1e-6)
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
        for ii, batch in enumerate(train_ds):
            loss_item = train(model, batch, device, optimizer)
            train_loss += loss_item  # model.total_loss.item()
            print(
                f"Epoch [{epoch+1}]|log10 loss:{np.log10(loss_item):.4f}|MSE Loss:{loss_item:.9f}|\r",
                end="",
            )

        # test
        with torch.no_grad():
            test_loss = test(model, test_ds, device, epoch, par_args.save_imag_path)

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
            f"Epoch [{epoch+1}/{par_args.num_epoch}]|Loss:{np.log10(train_loss/(ii+1)):.7f}|test loss:{np.log10(test_loss):.7f}|lr:{lr:.8f}|Time:{end_time-start_time:.2f}s|"
        )

    logging.info("finish training!")

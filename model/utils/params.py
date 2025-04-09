import argparse


def par():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--parameter_dim", type=int, default=4)
    parser.add_argument("--waveform_len", type=int, default=1024)
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hidden_layer", type=int, default=6)
    parser.add_argument("--num_epoch", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--load_check_point", type=str, default="False")
    parser.add_argument("--interpolation", type=str, default="True")
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--save_checkpoint_path", type=str, default=None)
    parser.add_argument("--save_imag_path", type=str, default=None)
    parser.add_argument("--save_log_path", type=str, default=None)

    args = parser.parse_args()
    return args


def str_to_bool(str_par):
    if str_par == "True":
        return True
    elif str_par == "False":
        return False
    else:
        raise ValueError("Must be True or False")

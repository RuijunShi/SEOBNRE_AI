python par_to_mergetime.py \
    --gpu_id 2\
    --parameter_dim 4\
    --waveform_len 1024\
    --encoder_dim 1024\
    --hidden_dim 1024\
    --num_epoch 50000\
    --batch_size 1024\
    --hidden_layer 4\
    --lr 1e-2\
    --interpolation True\
    --load_check_point False\
    --dataset_path ./dataset/waveformv_ecc_v6_spinz.h5\
    --model_checkpoint /workspace\
    --save_checkpoint_path ./check_point/sample_1024_4096Hz_merge_time_4par_4layer_L1loss.pth\
    --save_imag_path ./image/sample_1024_4096Hz_merge_time_4par_4layer_L1loss.png\
    --save_log_path ./log/sample_1024_4096Hz_merge_time_4par_4layer_L1loss.log

## 
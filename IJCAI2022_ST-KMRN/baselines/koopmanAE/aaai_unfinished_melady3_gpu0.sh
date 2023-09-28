CUDA_VISIBLE_DEVICES=0 python driver_multires.py --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --test_seq_diff 48 --keep_ratio 1.0 --target_res all --resolution_type agg --alpha 8 --lr 5e-4 --epochs 300 --batch 32 --batch_test 32 --steps 480 --steps_back 480 --bottleneck 128 --backward 1 --pred_steps 480 --seed 44
CUDA_VISIBLE_DEVICES=0 python driver_multires.py --data ../../data/nyc_taxi/sel_green --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --test_seq_diff 48 --keep_ratio 1.0 --target_res all --resolution_type agg --alpha 8 --lr 5e-4 --epochs 150 --batch 32 --batch_test 32 --steps 480 --steps_back 480 --bottleneck 128 --backward 1 --pred_steps 480 --seed 44
CUDA_VISIBLE_DEVICES=0 python driver_multires.py --data ../../data/solar_energy_10min --seq_in_len 1440 --seq_out_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1.0 --target_res all --resolution_type agg --alpha 8 --lr 5e-4 --epochs 10 --batch 32 --batch_test 32 --steps 432 --steps_back 432 --bottleneck 128 --backward 1 --pred_steps 432 --seed 43
CUDA_VISIBLE_DEVICES=0 python driver_multires.py --data ../../data/solar_energy_10min --seq_in_len 1440 --seq_out_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1.0 --target_res all --resolution_type agg --alpha 8 --lr 5e-4 --epochs 10 --batch 32 --batch_test 32 --steps 432 --steps_back 432 --bottleneck 128 --backward 1 --pred_steps 432 --seed 44

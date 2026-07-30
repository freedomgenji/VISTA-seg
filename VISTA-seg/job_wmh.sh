export CUDA_VISIBLE_DEVICES=1
python train.py \
    --batch_size 2 \
    --iter_per_epoch -1 \
    --num_epochs 500 \
    --savepath /mnt/data2/xxx/output/hetero/vista_seg_wmh_split_person2 \
    --shared_rep_method vista_seg \
    --use_reg_loss \
    --dataname wmh \
    --datapath /data1/xxx/datasets/Zhe_2_MRI/processed

python covid-optimize-inference.py \
    --cktpath checkpoints/exp3/RoPe_RoPeflen5_iter5_loss1_1_2_lr0.0001_bsz2_seed128/checkpoint_best.pt \
    --save_dir checkpoints/Nb_Ab_exp3-covid-RoPE-ckpts/exp3-sabdab-bsz64-checkpoint5.pt \
    --cluster abgnn/finetune/exp3-covabdab/cdrh3_split.txt \
    --topk 64 --epochs 10000


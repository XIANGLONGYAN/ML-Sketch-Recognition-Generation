python eval_ours.py \
  --gen_root /data/user/jackyan/CS3308/sketch_code/generation/results \
  --ckpt ./pretrain/QD414k.pkl \
  --device cuda:0 \
  --img_size 224 \
  --batch_size 64 \
  --use_branch img \
  --data_root ./Data \
  --out_cm confusion_matrix_gen.npy \
  --out_csv per_class_acc_gen.csv
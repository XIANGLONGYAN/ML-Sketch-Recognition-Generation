python eval_1.py \
  --gen_root /data/user/jackyan/CS3308/sketch_code/generation/results \
  --ckpt ./pretrain/QD414k.pkl \
  --device cuda:0 \
  --img_size 224 \
  --batch_size 64 \
  --use_logits img \
  --out_cm confusion_matrix_gen.npy \
  --out_csv per_class_acc_gen.csv



python eval_gen_images.py \
  --gen_root /data/user/jackyan/CS3308/sketch_code/generation/results \
  --ckpt ./pretrain/QD414k.pkl \
  --device cuda:0 \
  --class_list /data/user/jackyan/CS3308/sketch_code/generation/category_list.txt \
  --use_logits img

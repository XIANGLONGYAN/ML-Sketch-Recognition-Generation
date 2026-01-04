CUDA_VISIBLE_DEVICES=2,3,4,5,6 python -u Train.py


Move your model into the './pretrain' folder.
Then, rename the model as 'QD.pkl' (trained on QuickDraw) or 'QD414k.pkl' (trained on QuickDraw414k).
CUDA_VISIBLE_DEVICES=1,2 python -u Eval.py
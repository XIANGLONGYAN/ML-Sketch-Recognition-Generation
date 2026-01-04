import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# 导入自定义模块
from Dataset import get_dataloader
from Networks5 import net
from Hyper_params import hp
from metrics import AverageMeter, accuracy

# 设置随机种子
seed = 1010
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print("***********- READ DATA and processing -*************")
dataloader_Train, dataloader_Test, dataloader_Valid = get_dataloader()

print("***********- loading model -*************")
if len(hp.gpus) == 0:
    model = net()
elif len(hp.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
    model = net().cuda()
else:
    gpus_str = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    model = net().cuda()
    gpus_list = [i for i in range(len(hp.gpus))]
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

# 确定模型名称
if hp.Dataset == 'QuickDraw':
    model_name = 'QD'
elif hp.Dataset == 'QuickDraw414k':
    model_name = 'QD414k'
else:
    model_name = 'Custom'

print(f'Loading pretrain model: ./pretrain/{model_name}.pkl')
checkpoint = torch.load('./pretrain/' + model_name + '.pkl')['net_state_dict']
model.load_state_dict(checkpoint)
loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)

class weight_record():
    def __init__(self):
        self.weight_img_record = np.zeros([hp.categories])
        self.cat_num = np.zeros([hp.categories])

    def update(self, labels, weight):
        for i, label in enumerate(labels):
            # 将 tensor 转为标量
            w_val = weight[i][0].cpu().item() if torch.is_tensor(weight) else weight[i][0]
            self.weight_img_record[label] += w_val
            self.cat_num[label] += 1

    def calculate(self):
        # 避免除以 0
        self.weight_img_record = np.divide(self.weight_img_record, self.cat_num, 
                                          out=np.zeros_like(self.weight_img_record), 
                                          where=self.cat_num != 0)

w_r = weight_record()

class trainer:
    def __init__(self, loss_f, model):
        self.loss_f = loss_f
        self.model = model

    def valid_epoch(self, loader, name="Eval"):
        self.model.eval()
        loader = tqdm(loader, desc=f"Processing {name}")
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(loader):
            with torch.no_grad():
                imgs = batch['sketch_img']
                seqs = batch['sketch_points']
                labels = batch['sketch_label']
                
                if len(hp.gpus) > 0:
                    batch_imgs, batch_seqs, batch_labels = imgs.cuda(), seqs.cuda(), labels.cuda()
                else:
                    batch_imgs, batch_seqs, batch_labels = imgs, seqs, labels

                predicted, img_logsoftmax, seq_logsoftmax, cv_important = self.model(batch_imgs, batch_seqs)
                
                # 只有在测试集或验证集时更新权重记录（可选，这里保持原逻辑）
                if name == "Test":
                    w_r.update(labels, cv_important)
                
                # 收集混淆矩阵数据
                _, preds = torch.max(predicted, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

                loss = self.myloss(predicted, img_logsoftmax, seq_logsoftmax, batch_labels)
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))

        return top1.avg, top5.avg, losses.avg, all_labels, all_preds

    def myloss(self, predicted, img_ls, seq_ls, labels):
        mix_loss = self.loss_f(predicted, labels)
        img_loss = self.loss_f(img_ls, labels)
        seq_loss = self.loss_f(seq_ls, labels)
        return hp.mix_weight * mix_loss + hp.img_weight * img_loss + hp.seq_weight * seq_loss

    def analyze_results(self, y_true, y_pred, phase_name):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 1. 打印最容易混淆的前 10 对
        cm_temp = cm.copy()
        np.fill_diagonal(cm_temp, 0)
        indices = np.argsort(cm_temp.flatten())[::-1][:10]
        
        print(f"\n[{phase_name}] Top 10 Misclassified Pairs:")
        for idx in indices:
            t_idx, p_idx = idx // cm.shape[1], idx % cm.shape[1]
            if cm[t_idx, p_idx] > 0:
                print(f"  True: {t_idx} -> Pred: {p_idx} ({cm[t_idx, p_idx]} times)")

        # 2. 绘制并保存热力图
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title(f'Confusion Matrix - {phase_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'CM_{model_name}_{phase_name}.png', dpi=300)
        plt.close()

    def run(self, train_loader, val_loader, test_loader):
        # 按照你的要求，依次跑三个数据集
        datasets = [("Train", train_loader), ("Valid", val_loader), ("Test", test_loader)]
        
        summary = {}
        for name, loader in datasets:
            err1, err5, loss, y_true, y_pred = self.valid_epoch(loader, name)
            self.analyze_results(y_true, y_pred, name)
            summary[name] = (err1, err5, loss)
        
        # 权重统计持久化
        w_r.calculate()
        with open(model_name + '_weight_record.pkl', 'wb') as f:
            pickle.dump(w_r, f)
        
        # 最终打印对比
        print("\n" + "="*30 + " FINAL REPORT " + "="*30)
        for name, (e1, e5, l) in summary.items():
            print(f"{name:5s} | Loss: {l:.4f} | Top-1 Acc: {100-e1:.2f}% | Top-5 Acc: {100-e5:.2f}%")

# 执行
print('''***********- Evaluating All Datasets -*************''')
params_total = sum(p.numel() for p in model.parameters())
print("Model Parameters: %.2fM" % (params_total / 1e6))

Trainer = trainer(loss_f, model)
Trainer.run(dataloader_Train, dataloader_Valid, dataloader_Test)
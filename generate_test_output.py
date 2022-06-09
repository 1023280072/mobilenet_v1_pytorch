# 比赛网址：https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data?select=sample_submission.csv
# 不知道为什么，使用此工程产生的csv文件提交比赛之后，所得结果十分奇怪，我也不明白为什么会这样

import torch
import pandas as pd
from tqdm import tqdm
from dataset import DogvsCatDataset
from torch.utils.data import DataLoader

def main():
    net = torch.load('work_dir/checkpoints/epoch_240.pth')
    test_dataset = DogvsCatDataset('dogvscat/test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds = []
    i = 1
    for img, _ in tqdm(test_dataloader):
        img = img.cuda()
        pred = net(img)
        pred = torch.sigmoid(pred)
        pred = float(pred)
        pred = round(pred, 4)
        preds.append([i, pred])
        i += 1
    df = pd.DataFrame(preds, columns=['id', 'label'])
    df.to_csv('dogvscat/submit_epoch_240.csv', index=False)




if __name__ == '__main__':
    main()

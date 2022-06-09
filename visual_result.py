import torch
import os
import matplotlib.pyplot as plt
from dataset import DogvsCatDataset
from PIL import Image

test_images_dir = 'dogvscat/test'

def main():
    images_names = os.listdir(test_images_dir)
    test_dataset = DogvsCatDataset(test_images_dir)
    net = torch.load('work_dir/checkpoints/epoch_240.pth')

    for image_name in images_names:
        image_path = os.path.join(test_images_dir, image_name)
        ori_img = Image.open(image_path)
        img = test_dataset.transform(ori_img)
        img = img.cuda()
        img = img.unsqueeze(0)
        pred = net(img)
        pred = torch.sigmoid(pred)
        if pred > 0.5:
            print('dog')
        elif pred <= 0.5:
            print('cat')
        plt.imshow(ori_img)
        plt.show()
        os.system('pause')



if __name__ == '__main__':
    main()

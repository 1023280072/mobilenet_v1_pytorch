import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

Mean = [0.41651208, 0.45481538, 0.48827952]
Std = [0.22509782, 0.22477485, 0.22919002]
# class:
#   cat: 0
#   dog: 1
class DogvsCatDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.imgs_names = os.listdir(imgs_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(Mean, Std)
        ])

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, index):
        img_name = self.imgs_names[index]
        img_path = os.path.join(self.imgs_dir, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        img_name = img_name.split('.')[0]
        if img_name == 'cat':
            label = 0
        elif img_name == 'dog':
            label = 1
        else:
            label = -1
        return img, label




if __name__ == '__main__':
    imgs_dir = 'dogvscat/train'
    dataset = DogvsCatDataset(imgs_dir)
    dataset.__getitem__(10)
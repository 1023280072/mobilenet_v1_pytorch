# 参考：https://blog.csdn.net/haseetxwd/article/details/89236984?ops_request_misc=&request_id=&biz_id=102&utm_term=python%20%E7%BB%9F%E8%AE%A1%E5%9B%BE%E5%83%8F%E5%9D%87%E5%80%BC%E5%92%8C%E6%A0%87%E5%87%86%E5%B7%AE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-89236984.nonecase&spm=1018.2226.3001.4187
# PIL读入图片颜色顺序为RGB
# opencv则为BGR
# 所以在填入dataset.py文件时应将结果颠倒

import cv2, os, argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='dogvscat/train', type=str)
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    img_filenames = os.listdir(opt.dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(opt.dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])

if __name__ == '__main__':
    main()
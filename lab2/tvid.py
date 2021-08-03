import os
from torch.utils import data
import transforms
import concurrent.futures
import numpy as np
from PIL import Image
classes = ['bird', 'car', 'dog', 'lizard', 'turtle']
classDict = dict([(classes[i], i) for i in range(len(classes))])


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.

    def __init__(self, root, mode):
        self.path = root + '/'
        self.mode = mode
        self.classSize = 180
        self.processor = transforms.Compose([transforms.LoadImage(), transforms.ToTensor(),
                                             transforms.Resize(600)])  #
        self.sample = []
        files = os.listdir(self.path)
        files = [files[i:i + 2] for i in range(0, len(files), 2)]
        self.labelDict = classDict
        self.label = classes
        # 可以考虑做正则匹配，不过既然有明确的命名规则就不搞了
        # 两两匹配有出错风险，这里暂不考虑
        with concurrent.futures.ThreadPoolExecutor(min(len(files) // 2, 5)) as e:
            sampleByClass = e.map(self._get_class, files)
        for s in sampleByClass:
            self.sample += s
        self.classNum = len(self.labelDict)

    def getLabels(self):
        return self.labelDict

    def _get_class(self, info):
        """
        param:
            info: [type/folderName, annotationName]
        return:
            [[picPath, [label, bboxFig]],.....]
        """
        label = self.labelDict[info[0]]

        picpath = self.path + info[0] + '/'
        pic = os.listdir(picpath)
        with open(self.path + info[1], 'r') as fp:
            info = fp.readlines()
        info[-1] = info[-1] if info[-1][-1] == '\n' else info[-1] + '\n'
        info = [i[:-1].split(sep=' ') for i in info]
        for i in range(len(info)):
            info[i] = [int(fig) for fig in info[i]]

        picinfo = [[picpath + pic[i], info[i][1:], label] for i in range(self.classSize)]
        return picinfo

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        img, bbox, label = self.sample[item]
        # label = self.label[label]
        X = self.processor(img, bbox)
        return X, label

    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root='./tiny_vid', mode='train')
    x, y = dataset[0]
    print(1)

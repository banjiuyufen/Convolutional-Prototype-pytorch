"""
functions:
    1.CNN model
    2.character classifier

author:
    ychen
2021.10.8
"""



import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# 定义模型
class HccrNet_cpn(nn.Module):
    def __init__(self, num_classes=7356):
        super(HccrNet_cpn, self).__init__()
        self.features_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=150, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(150,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(200,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(200,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_3 = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=250, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(250,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=250, out_channels=300, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(300,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(300,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_4 = nn.Sequential(
            nn.Conv2d(in_channels=300, out_channels=350, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(350,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=350, out_channels=400, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(400,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=400, out_channels=400, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(400,0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 400 * 4 * 4, out_features=900),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=900, out_features=200),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.0),
            #nn.Linear(in_features=200, out_features=num_classes+1)
        )
        # 采用全0初始化原型
        self.centers = nn.Parameter(torch.zeros([num_classes + 1, 200], dtype=torch.float))
        
        # 采用(0,1)均匀分布初始化原型
        #self.centers = nn.Parameter(torch.rand(num_classes + 1, 200))
        
        # 采用高斯分布初始化原型
        #self.centers = nn.Parameter(torch.randn(num_classes + 1, 200))

        # 初始化OVA损失函数半径
        #self.r = nn.Parameter(torch.ones(num_classes+1,1))

    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = x.view(-1, 400 * 4 * 4)
        FCfeat = self.classifier(x) # ndim = 200
        return FCfeat, self.centers


class HccrNet_no_BatchNorm(nn.Module):
    def __init__(self, num_classes=7356):
        super(HccrNet_no_BatchNorm, self).__init__()
        self.features_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_3 = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=250, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=250, out_channels=300, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_4 = nn.Sequential(
            nn.Conv2d(in_channels=300, out_channels=350, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=350, out_channels=400, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=400, out_channels=400, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400 * 4 * 4, out_features=900),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=900, out_features=200),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.0),
            # nn.Linear(in_features=200, out_features=num_classes+1)
        )
        # 采用全0初始化原型
        self.centers = nn.Parameter(torch.zeros([num_classes, 200], dtype=torch.float))

        # 采用(0,1)均匀分布初始化原型
        # self.centers = nn.Parameter(torch.rand(num_classes + 1, 200))

        # 采用高斯分布初始化原型
        #self.centers = nn.Parameter(torch.randn(num_classes + 1, 200))

    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = x.view(-1, 400 * 4 * 4)
        FCfeat = self.classifier(x)  # ndim = 200
        return FCfeat, self.centers

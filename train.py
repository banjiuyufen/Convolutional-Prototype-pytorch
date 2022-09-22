import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import CPN_Net
import Data.data_set_2 as dataset
import time
from tensorboardX import SummaryWriter
logdir = '/home/log/2022_9_21_cpn/'
writer = SummaryWriter(log_dir=logdir)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    return model


root = 'Data/' 

batch_size = 512
normalize = transforms.Normalize((0.5,), (0.5,))

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
text_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


train_data = dataset.MyDataset(txt=root + 'train.txt', transform=train_transforms)
test_data = dataset.MyDataset(txt=root + 'test.txt', transform=text_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

num_classes=10
#model = CPN_Net.HccrNet(num_classes)
model = CPN_Net.HccrNet_no_BatchNorm(num_classes)
model = weights_init(model)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)  #
#optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9,0.999), eps=1e-08)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 15, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('Lets use', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)

def del_tensor_ele(arr, index):
    arr1 = arr[0:index,:]
    arr2 = arr[index + 1:,:]
    return torch.cat((arr1, arr2), dim=0)

def softmax_loss(logits, labels):
    labels = labels
    criterion = nn.CrossEntropyLoss()
    cross_entropy = criterion(logits, labels)
    return torch.mean(cross_entropy)


def distance(features, centers):
    f_2 = features.pow(2).sum(dim=1, keepdim=True)
    c_2 = centers.pow(2).sum(dim=1, keepdim=True)
    dist = f_2 - 2 * torch.matmul(features, centers.transpose(0, 1)) + c_2.transpose(0, 1)
    return dist


def dce_loss(features, labels, centers, T):
    dist = distance(features, centers)
    logits = -dist / T
    mean_loss = softmax_loss(logits, labels)
    return mean_loss
    
def pl_loss(features, labels, centers):
    batch_num = features.shape[0]
    batch_centers = torch.index_select(centers, 0, labels)
    dis = features - batch_centers
    return torch.div(l2_loss(dis), batch_num)
    

def l2_loss(a):
    l2_dis = torch.pow(a,2).sum()
    return l2_dis/2

def predict(features, centers):
    dist = distance(features, centers)
    prediction = torch.argmin(dist, dim=1)
    return prediction

def evalution(features, labels, centers):
    dist = distance(features, centers)
    prediction = torch.argmin(dist, dim=1)
    correct = torch.equal(prediction, labels)
    return torch.sum(correct)

def train(epoch):
    print("---------- train-epoch{} --------------".format(epoch))
    losses = AverageMeter()
    correct = 0
    total = 0
    model.train()
    for batch_idx, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features, centers = model(images)
        loss1 = dce_loss(features, labels, centers, 1)
        loss2 = pl_loss(features, labels, centers)
        loss = loss1 + 0.0001*loss2
        losses.update(loss.item(), labels.size(0))
        loss.backward()
        optimizer.step()
        prediction = predict(features,centers)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
        if batch_idx % 30 == 0:
            print(
                    '[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())),
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        optimizer.param_groups[0]["lr"],
                        loss=losses,
                    ))
            print('Accuracy on train set: %.6f %% [%d/%d]' % (100 * correct / total, correct, total))
            running_loss = 0.000
        writer.add_scalar('train_loss_cpn',losses.avg,batch_idx)

def test(epoch):
    print("---------- test-epoch{} --------------".format(epoch))
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad(): 
        for batch_idx, data in enumerate(test_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)           
            features,centers = model(images)           
            total += labels.size(0)                    
            evel_correct = predict(features,centers)
            correct += (evel_correct == labels).sum().item()
            
            loss1 = dce_loss(features, labels, centers, 1)
            loss2 = pl_loss(features, labels, centers)
            loss = loss1 + 0.0001*loss2
            test_loss += loss.item()
            print('[{}] Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())),
                    epoch,
                    batch_idx * len(data),
                    len(test_loader.dataset),
                    100. * batch_idx / len(test_loader),
                    test_loss / labels.size(0)
                ))
            print('Accuracy on test set: %.6f %% [%d/%d]' % (100 * correct / total, correct, total))
    return 100 * correct / total


if __name__ == '__main__':
    for epoch in range(80):
        
        train(epoch)
        
        test_acc = test(epoch)
       
        scheduler.step()
        torch.save({'model': model.state_dict()},
                   "/" + str(epoch)  + ".pth")
 

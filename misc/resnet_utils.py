import torch
import torch.nn as nn
import torch.nn.functional as F


class myResnet(nn.Module):  #
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze() # torch.Size([2048])

        #  feature map 7x7 and 14x14
        att1 = F.adaptive_avg_pool2d(x, [7, 7]).squeeze().permute(1, 2, 0)
        att1=att1.reshape(att1.shape[0] * att1.shape[1], att1.shape[2])
        att2 = F.adaptive_avg_pool2d(x, [14, 14]).squeeze().permute(1, 2, 0)  # 14, 14, 2048
        att2=att2.reshape(att2.shape[0] * att2.shape[1], att2.shape[2])
        att = torch.cat([att1, att2], 0)

        return fc, att

import torch.nn as nn


class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ELU(1.0)
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.drop1 = nn.Dropout(0.3)
        conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ELU(1.0)
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.drop2 = nn.Dropout(0.3)
        conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ELU(1.0)
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.drop3 = nn.Dropout(0.3)
        conv_layers += [self.conv3, self.relu3, self.bn3, self.drop3]

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ELU(1.0)
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.drop4 = nn.Dropout(0.3)
        conv_layers += [self.conv4, self.relu4, self.bn4, self.drop4]

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ELU(1.0)
        self.bn5 = nn.BatchNorm2d(128)
        nn.init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.drop5 = nn.Dropout(0.3)
        conv_layers += [self.conv5, self.relu5, self.bn5, self.drop5]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
                
        self.lin1 = nn.Linear(in_features=128, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=1)
        
        self.relu = nn.ELU(1.0)

        self.conv = nn.Sequential(*conv_layers)
 

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x.squeeze(1)
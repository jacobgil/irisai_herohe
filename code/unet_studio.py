import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, factor=0.25, kernel_size=3, input_size=3, num_outputs=3):
        super(Unet, self).__init__()

        self.num_outputs = num_outputs

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = \
            nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout = nn.Dropout2d(p=0.2, inplace=True)
        self.bn1_0 = nn.BatchNorm2d(int(32*factor))
        self.bn1_1 = nn.BatchNorm2d(int(32*factor))
        self.bn2_0 = nn.BatchNorm2d(int(64*factor))
        self.bn2_1 = nn.BatchNorm2d(int(64*factor))
        self.bn3_0 = nn.BatchNorm2d(int(128*factor))
        self.bn3_1 = nn.BatchNorm2d(int(128*factor))
        self.bn4_0 = nn.BatchNorm2d(int(256*factor))
        self.bn4_1 = nn.BatchNorm2d(int(256*factor))
        self.bn5_0 = nn.BatchNorm2d(int(512*factor))
        self.bn5_1 = nn.BatchNorm2d(int(512*factor))
        self.bn6_0 = nn.BatchNorm2d(int(256*factor))
        self.bn6_1 = nn.BatchNorm2d(int(256*factor))
        self.bn7_0 = nn.BatchNorm2d(int(128*factor))
        self.bn7_1 = nn.BatchNorm2d(int(128*factor))
        self.bn8_0 = nn.BatchNorm2d(int(64*factor))
        self.bn8_1 = nn.BatchNorm2d(int(64*factor))
        self.bn9_0 = nn.BatchNorm2d(int(32*factor))
        self.bn9_1 = nn.BatchNorm2d(int(32*factor))

        self.conv1_0 = nn.Conv2d(input_size, int(32*factor), kernel_size=kernel_size, \
            padding=kernel_size//2)
        self.conv1_1 = nn.Conv2d(int(32*factor), int(32*factor), kernel_size=kernel_size, \
            padding=kernel_size//2)
        
        self.conv2_0 = nn.Conv2d(int(32*factor), int(64*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2_1 = nn.Conv2d(int(64*factor), int(64*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.conv3_0 = nn.Conv2d(int(64*factor), int(128*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3_1 = nn.Conv2d(int(128*factor), int(128*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.conv4_0 = nn.Conv2d(int(128*factor), int(256*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4_1 = nn.Conv2d(int(256*factor), int(256*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.conv5_0 = nn.Conv2d(int(256*factor), int(512*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv5_1 = nn.Conv2d(int(512*factor), int(512*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.up6 = nn.ConvTranspose2d(int(512*factor), int(256*factor), \
            kernel_size=2, stride=2, padding=0)
        self.conv6_0 = nn.Conv2d(int(512*factor), int(256*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv6_1 = nn.Conv2d(int(256*factor), int(256*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.up7 = nn.ConvTranspose2d(int(256*factor), int(128*factor), \
            kernel_size=2, stride=2, padding=0)
        self.conv7_0 = nn.Conv2d(int(256*factor), int(128*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv7_1 = nn.Conv2d(int(128*factor), int(128*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.up8 = nn.ConvTranspose2d(int(128*factor), int(64*factor), \
            kernel_size=2, stride=2, padding=0)
        self.conv8_0 = nn.Conv2d(int(128*factor), int(64*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv8_1 = nn.Conv2d(int(64*factor), int(64*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.up9 = nn.ConvTranspose2d(int(64*factor), int(32*factor), \
            kernel_size=2, stride=2, padding=0)
        self.conv9_0 = nn.Conv2d(int(64*factor), int(32*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)
        self.conv9_1 = nn.Conv2d(int(32*factor), int(32*factor), \
            kernel_size=kernel_size, padding=kernel_size//2)

        self.output_conv = nn.Conv2d(int(32*factor), num_outputs, kernel_size=1)
                
    
    def decoder(self, conv5, conv4, conv3, conv2, conv1):
        up6 = torch.cat((self.up6(conv5), conv4), 1)
        conv6 = self.bn6_0(self.relu(self.conv6_0(up6))) #dataset A
        conv6 = self.bn6_1(self.relu(self.conv6_1(conv6)))
                
        up7 = torch.cat((self.up7(conv6), conv3), 1)
        conv7 = self.bn7_0(self.relu(self.conv7_0(up7)))
        conv7 = self.bn7_1(self.relu(self.conv7_1(conv7)))

        up8 = torch.cat((self.up8(conv7), conv2), 1)
        conv8 = self.bn8_0(self.relu(self.conv8_0(up8)))
        conv8 = self.bn8_1(self.relu(self.conv8_1(conv8)))

        up9 = torch.cat((self.up9(conv8), conv1), 1)
        conv9 = self.bn9_0(self.relu(self.conv9_0(up9)))
        conv9 = self.dropout(conv9)
        conv9 = self.bn9_1(self.relu(self.conv9_1(conv9)))

        return self.output_conv(conv9)
    
    def encoder(self, x):
        conv1 = self.bn1_0(self.relu(self.conv1_0(x)))
        conv1 = self.bn1_1(self.relu(self.conv1_1(conv1)))

        conv2 = self.maxpool(conv1)
        conv2 = self.bn2_0(self.relu(self.conv2_0(conv2)))
        conv2 = self.bn2_1(self.relu(self.conv2_1(conv2)))

        conv3 = self.maxpool(conv2)
        conv3 = self.bn3_0(self.relu(self.conv3_0(conv3)))
        conv3 = self.bn3_1(self.relu(self.conv3_1(conv3)))

        conv4 = self.maxpool(conv3)
        conv4 = self.bn4_0(self.relu(self.conv4_0(conv4)))
        conv4 = self.bn4_1(self.relu(self.conv4_1(conv4)))

        conv5 = self.maxpool(conv4)
        conv5 = self.bn5_0(self.relu(self.conv5_0(conv5)))
        conv5 = self.dropout(conv5)
        conv5 = self.bn5_1(self.relu(self.conv5_1(conv5)))
        conv5 = self.dropout(conv5)
        
        return conv5, conv4, conv3, conv2, conv1
        
    def forward(self, x):
        encoder_output, conv4, conv3, conv2, conv1 = self.encoder(x)
        decoder_output = self.decoder(encoder_output, conv4, conv3, conv2, conv1)        
        return decoder_output
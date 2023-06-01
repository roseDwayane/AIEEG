import torch
from torch import nn
import numpy as np

__all__ = ['UNet', 'NestedUNet3', 'NestedUNet4', 'NestedUNet5']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, ks=7):
        super().__init__()
        padding = int((ks - 1) / 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=ks, padding=padding)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=ks, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MultiHeadAttentation(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadAttentation, self).__init__()
        self.d = d
        self.n_heads = n_heads
        self.q_mappings = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d, d) for _ in range(self.n_heads)])
        self.o_mappings = nn.Linear(d * n_heads, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result_head = []
        # multi-head
        for head in range(self.n_heads):
            q_mapping = self.q_mappings[head]
            k_mapping = self.k_mappings[head]
            v_mapping = self.v_mappings[head]

            q, k, v = q_mapping(sequences), k_mapping(sequences), v_mapping(sequences)
            k = torch.transpose(k, 1, 2)
            a = torch.matmul(q, k)
            attention = self.softmax(a / (self.d ** 0.5))
            #print("attention: ", attention.shape)
            o = torch.matmul(attention, v)
            result_head.append(o)

        #print("result_head: ", len(result_head))
        o = torch.cat(result_head, axis=-1)
        #print("output1: ", o.shape)
        o = self.o_mappings(o)
        #print("output2: ", o.shape)

        return o

class AttentionBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(AttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mha = MultiHeadAttentation(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mha(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class UNetpp3_Transformer(nn.Module):
    def __init__(self, num_classes, input_channels=30, n_patches=7, n_blocks=2, n_heads=2):
        super(UNetpp3_Transformer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool1d(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.positional_embeddings = self.get_positional_embeddings(30, 1024).to(device)

        self.blocks0_0 = nn.ModuleList([AttentionBlock(1024, n_heads) for _ in range(n_blocks)])
        self.blocks1_0 = nn.ModuleList([AttentionBlock(512, n_heads) for _ in range(n_blocks)])
        self.blocks2_0 = nn.ModuleList([AttentionBlock(256, n_heads) for _ in range(n_blocks)])
        self.blocks3_0 = nn.ModuleList([AttentionBlock(128, n_heads) for _ in range(n_blocks)])

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

        # if self.deep_supervision:
        self.final1 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):

        new_input = input + self.positional_embeddings

        for block in self.blocks0_0:
            x0_0 = block(new_input)
        x0_0 = self.conv0_0(x0_0)

        for block in self.blocks1_0:
            x1_0 = block(self.pool(x0_0))
        x1_0 = self.conv1_0(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #print("input:", input.shape)
        #print("x0_0: ", x0_0.shape)
        #print("pool: ", self.pool(x0_0).shape)
        #print("x1_0: ", x1_0.shape)
        #print("up:   ", self.up(x1_0).shape)
        #print("cat:  ", torch.cat([x0_0, self.up(x1_0)], 1).shape)
        #print("x0_1: ", x0_1.shape)
        for block in self.blocks2_0:
            x2_0 = block(self.pool(x1_0))
        x2_0 = self.conv2_0(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        for block in self.blocks3_0:
            x3_0 = block(self.pool(x2_0))
        x3_0 = self.conv3_0(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        #if self.deep_supervision:
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        return output1, output2, output3

    def get_positional_embeddings(self, channel, time_len):
        result = torch.ones(channel, time_len)
        for i in range(channel):
            for j in range(time_len):
                result[i][j] = np.sin(j / (10000 ** (i / channel))) if i % 2 == 0 else np.cos(j / (10000 ** ((i - 1) / channel)))
        return result



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool1d(2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        # input_channel => 32; 32 => 64; 64=>128; 128=>256
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class NestedUNet3(nn.Module):
    def __init__(self, num_classes, input_channels=30, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool1d(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])

        #if self.deep_supervision:
        self.final1 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        #else:
            #self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #print("input:", input.shape)
        #print("x0_0: ", x0_0.shape)
        #print("pool: ", self.pool(x0_0).shape)
        #print("x1_0: ", x1_0.shape)
        #print("up:   ", self.up(x1_0).shape)
        #print("cat:  ", torch.cat([x0_0, self.up(x1_0)], 1).shape)
        #print("x0_1: ", x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        #if self.deep_supervision:
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        return output1, output2, output3
        """
        else:
            output = self.final(x0_4)
            return output
        """

class NestedUNet4(nn.Module):
    def __init__(self, num_classes, input_channels=30, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool1d(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        #if self.deep_supervision:
        self.final1 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        #else:
            #self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #print("input:", input.shape)
        #print("x0_0: ", x0_0.shape)
        #print("pool: ", self.pool(x0_0).shape)
        #print("x1_0: ", x1_0.shape)
        #print("up:   ", self.up(x1_0).shape)
        #print("cat:  ", torch.cat([x0_0, self.up(x1_0)], 1).shape)
        #print("x0_1: ", x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        #if self.deep_supervision:
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        return output1, output2, output3, output4
        """
        else:
            output = self.final(x0_4)
            return output
        """

class NestedUNet5(nn.Module):
    def __init__(self, num_classes, input_channels=30, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool1d(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_2 = VGGBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])

        #if self.deep_supervision:
        self.final1 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        self.final5 = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)
        #else:
            #self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #print("input:", input.shape)
        #print("x0_0: ", x0_0.shape)
        #print("pool: ", self.pool(x0_0).shape)
        #print("x1_0: ", x1_0.shape)
        #print("up:   ", self.up(x1_0).shape)
        #print("cat:  ", torch.cat([x0_0, self.up(x1_0)], 1).shape)
        #print("x0_1: ", x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        #if self.deep_supervision:
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output5 = self.final5(x0_5)
        return output1, output2, output3, output4, output5
        """
        else:
            output = self.final(x0_4)
            return output
        """

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import data_util
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=50, help="training epochs")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=128, help="dimension of embedding")
parser.add_argument("--hidden_layer", type=list, default=[128, 64, 32], help="dimension of each hidden layer")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--model_path", type=str, default="model")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

cudnn.benchmark = True


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation

        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class SqEx(nn.Module):
    def __init__(self, n_features, reduction=2):
        super(SqEx, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class ResBlockSqEx(nn.Module):
    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()
        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.sqex = SqEx(n_features)

    def forward(self, x):
        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))
        y = self.sqex(y)
        y = torch.add(x, y)
        return y


class GlobalLocal(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GlobalLocal, self).__init__()
        dilations = [1, 4, 8, 9]
        self.channel_size = out_ch
        self.kernel_size = 2
        self.strides = 2

        self.local_signal = nn.Sequential(
            nn.Conv2d(in_ch, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            ResBlockSqEx(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
        )

        self.plane_asp = 32

        self.dilated_layers = nn.ModuleList([
            ASPP_module(self.plane_asp, self.plane_asp, dilation=d) for d in dilations
        ])

        self.res_blocks = nn.ModuleList([
            ResBlockSqEx(self.plane_asp) for _ in dilations
        ])

        self.reshape = nn.Conv2d(160, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.local_signal(x)
        low_level = x

        features = [dilated(x) for dilated in self.dilated_layers]
        features = [self.res_blocks[i](f) if i in [1, 2] else f for i, f in enumerate(features)]

        x = torch.cat(features, dim=1)
        x = torch.cat((x, low_level), dim=1)
        x = self.reshape(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
        )
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


# class ConvNCF(nn.Module):
#     def __init__(self, user_count, item_count, embedding_size=64, channel_size=32, heads=4, num_transformer_layers=2):
#         super(ConvNCF, self).__init__()
#         self.user_count = user_count
#         self.item_count = item_count
#         self.embedding_size = embedding_size
#         self.channel_size = channel_size
#
#         self.P = nn.Embedding(self.user_count, self.embedding_size)
#         self.Q = nn.Embedding(self.item_count, self.embedding_size)
#         self.transformer = nn.Sequential(
#             *[TransformerBlock(embedding_size, heads) for _ in range(num_transformer_layers)]
#         )
#         self.cnn = GlobalLocal(1, self.channel_size)
#         self.fc = nn.Linear(self.channel_size, 1)
#
#     def forward(self, user_ids, item_ids):
#         user_embeddings = self.P(user_ids)
#         item_embeddings = self.Q(item_ids)
#
#         interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
#         interaction_map = interaction_map.view((-1, self.embedding_size, self.embedding_size))
#
#         interaction_map = self.transformer(interaction_map)
#         interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))
#
#         feature_map = self.cnn(interaction_map)
#         feature_vec = feature_map.view((-1, self.channel_size))
#
#         prediction = self.fc(feature_vec)
#         prediction = prediction.view((-1))
#         return prediction

class ConvNCF(nn.Module):
    def __init__(self, user_count, item_count, embedding_size=64, channel_size=32, heads=4, num_transformer_layers=2):
        super(ConvNCF, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.embedding_size = embedding_size
        self.channel_size = channel_size

        self.P = nn.Embedding(self.user_count, self.embedding_size)
        self.Q = nn.Embedding(self.item_count, self.embedding_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding_size, heads) for _ in range(num_transformer_layers)]
        )
        self.cnn = GlobalLocal(1, self.channel_size)

        # We will set this after checking the size of the feature_map in the forward pass
        self.fc = None

    def forward(self, user_ids, item_ids):
        batch_size = user_ids.size(0)  # Get the batch size

        user_embeddings = self.P(user_ids)
        item_embeddings = self.Q(item_ids)

        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
        interaction_map = interaction_map.view((batch_size, self.embedding_size, self.embedding_size))

        interaction_map = self.transformer(interaction_map)
        interaction_map = interaction_map.view((batch_size, 1, self.embedding_size, self.embedding_size))

        feature_map = self.cnn(interaction_map)

        # Compute the size of the flattened feature_map
        feature_map_size = feature_map.size(1) * feature_map.size(2) * feature_map.size(3)
        feature_vec = feature_map.view((batch_size, feature_map_size))  # Flatten the feature map

        # Define the fully connected layer if not defined yet
        if self.fc is None:
            self.fc = nn.Linear(feature_map_size, 1).to(feature_map.device)

        prediction = self.fc(feature_vec)
        prediction = prediction.view((-1))
        return prediction


if __name__ == "__main__":
    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    model = ConvNCF(user_num, item_num, args.embedding_dim)
    model.to(device=args.device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.float().to(device=args.device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.device)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'ConvNCF.pth'))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn


import data_util
import evaluate
from MLP import ConvNCF

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=50, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimension of embedding")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--model_path", type=str, default="data")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

cudnn.benchmark = True


class GMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim):
        super(GMF, self).__init__()
        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)

        self.predict_layer = nn.Linear(embedding_dim, 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        output = embed_user * embed_item
        prediction = self.predict_layer(output)
        return prediction.view(-1)


class CombinedModel(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, channel_size=32, heads=4, num_transformer_layers=2):
        super(CombinedModel, self).__init__()
        self.gmf = GMF(user_num, item_num, embedding_dim)
        self.convncf = ConvNCF(user_num, item_num, embedding_dim, channel_size, heads, num_transformer_layers)

    def forward(self, user, item):
        gmf_prediction = self.gmf(user, item)
        convncf_prediction = self.convncf(user, item)

        # Combine predictions
        combined_prediction = gmf_prediction + convncf_prediction
        return combined_prediction


if __name__ == "__main__":
    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    model = CombinedModel(user_num, item_num, args.embedding_dim)
    model.to(device=args.device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
                torch.save(model.state_dict(), os.path.join(args.model_path, 'CombinedModel.pth'))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))

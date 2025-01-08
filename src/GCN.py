import json
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

# 数据集参数
# num_graphs = 30
# max_nodes_per_graph = 100
max_node_features = 768
num_classes = 15
# 模型参数
num_each_size = 32

""" # 固定GCN模型
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 全局池化层
        x = global_mean_pool(x, batch)
        # 全连接层分类
        x = self.fc(x)
        return F.log_softmax(x, dim=1) """

# K层GCN模型


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, 64))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(64, 64))
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 图卷积层
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        # 全局池化层
        x = global_mean_pool(x, batch)
        # 全连接层分类
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 生成数据


def generate_example_data(num_graphs, max_nodes_per_graph, max_node_features, num_classes):
    subgraph_adj_list = []
    subgraph_features_list = []
    labels = []
    for _ in range(num_graphs):
        # 生成随机大小的邻接矩阵，矩阵
        num_nodes = np.random.randint(5, max_nodes_per_graph + 1)
        adj = np.random.randint(0, 2, (num_nodes, num_nodes))
        np.fill_diagonal(adj, 0)
        # 生成固定大小的特征矩阵，矩阵
        features = np.random.rand(num_nodes, max_node_features)
        # 生成一个随机标签值，int
        label = np.random.randint(0, num_classes)
        # 邻接矩阵列表，特征矩阵列表，标签值列表
        # print(type(adj))
        # print(type(features))
        # print(type(label))
        # print(type(subgraph_adj_list))
        # print(type(subgraph_features_list))
        # print(type(labels))
        subgraph_adj_list.append(adj)
        subgraph_features_list.append(features)
        labels.append(label)
    return subgraph_adj_list, subgraph_features_list, labels

# 包装数据


def create_subgraph_data(subgraph_adj, subgraph_features, label):
    edge_index_array = np.array(subgraph_adj.nonzero())
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)
    x = torch.tensor(subgraph_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


""" # 数据集
subgraph_adj_list, subgraph_features_list, labels = generate_example_data(num_graphs, max_nodes_per_graph, max_node_features, num_classes)
# 训练集、验证集、测试集
adj_train, adj_temp, feat_train, feat_temp, labels_train, labels_temp = train_test_split(
    subgraph_adj_list, subgraph_features_list, labels, test_size=0.4, random_state=42)
adj_val, adj_test, feat_val, feat_test, labels_val, labels_test = train_test_split(
    adj_temp, feat_temp, labels_temp, test_size=0.5, random_state=42)
# print(type(adj_train))
# print(type(adj_test))
# print(type(adj_val)) """


def get_data_dir(file_dir):
    file_list = os.listdir(file_dir)
    num = 0
    adj = []
    feat = []
    labels = []
    for file in file_list:
        if not file.endswith('.pkl'):
            continue
        file_path = os.path.join(file_dir, file)
        file_path_json = f"{file_path}_encoded.json"
        # file_path_json = f"{file_path}.json"
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        with open(file_path_json, 'r', encoding='utf-8') as file_json:
            data_json = json.load(file_json)
        num += 1
        data["conn_array"] = np.array(data["conn_array"])
        data["conn_array"][data["conn_array"] != 0] = 1
        data_json['feature'] = np.array(data_json['feature'])
        if not (int(data_json['feature'].shape[0]) == int(data['conn_array'].shape[0])):
            continue
        adj.append(data['conn_array'])
        feat.append(data_json['feature'])
        labels.append(data_json['label'])
    # print(set(labels))
    return adj, feat, labels, num


def get_data(graph_dir):
    # 读取数据
    # 训练集、验证集、测试集

    # subgraph_adj_list, subgraph_features_list, labels, num_val = get_data_dir(graph_dir)
    # adj_train, adj_temp, feat_train, feat_temp, labels_train, labels_temp = train_test_split(
    # subgraph_adj_list, subgraph_features_list, labels, test_size=0.4, random_state=42)
    # adj_val, adj_test, feat_val, feat_test, labels_val, labels_test = train_test_split(
    # adj_temp, feat_temp, labels_temp, test_size=0.5, random_state=42)

    train_dir = os.path.join(graph_dir, 'train')
    test_dir = os.path.join(graph_dir, 'test')
    val_dir = os.path.join(graph_dir, 'val')
    adj_train, feat_train, labels_train, num_train = get_data_dir(train_dir)
    adj_test, feat_test, labels_test, num_test = get_data_dir(test_dir)
    adj_val, feat_val, labels_val, num_val = get_data_dir(val_dir)
    print(f'训练集Train:{num_train}')
    print(f'验证集Val:{num_val}')
    print(f'测试集Test:{num_test}')

    # 训练集列表、验证集列表、测试集列表
    train_data_list = [create_subgraph_data(
        adj, feat, label) for adj, feat, label in zip(adj_train, feat_train, labels_train)]
    val_data_list = [create_subgraph_data(
        adj, feat, label) for adj, feat, label in zip(adj_val, feat_val, labels_val)]
    test_data_list = [create_subgraph_data(
        adj, feat, label) for adj, feat, label in zip(adj_test, feat_test, labels_test)]
    return train_data_list, val_data_list, test_data_list


def train(graph_dir, epochs, learn_rate, num_layers, model_dir):
    train_data_list, val_data_list, test_data_list = get_data(graph_dir)
    # 训练集batch、验证集batch和测试集batch
    train_loader = DataLoader(
        train_data_list, batch_size=num_each_size, shuffle=True)
    val_loader = DataLoader(
        val_data_list, batch_size=num_each_size, shuffle=False)
    test_loader = DataLoader(
        test_data_list, batch_size=num_each_size, shuffle=False)
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(num_node_features=max_node_features,
                num_classes=num_classes,
                num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # 训练，验证
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, f'best_model_{learn_rate}_{num_layers}.pth')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += F.nll_loss(out, batch.y, reduction='sum').item()
        val_loss /= len(val_data_list)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            # print(f'Epoch{epoch+1:3d} | Train_loss:{loss.item():.6f} | Val_loss: {val_loss:.6f} |')
    # 测试
    model.load_state_dict(torch.load(best_model_path))
    print(best_model_path)
    model.eval()
    correct = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    all_preds = []
    all_labels = []
    ranks = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            _, top5_pred = out.topk(5, dim=1)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            correct_top1 += (top5_pred[:, 0] == batch.y).sum().item()
            correct_top3 += (top5_pred[:, :3] ==
                             batch.y.unsqueeze(1)).sum().item()
            correct_top5 += (top5_pred == batch.y.unsqueeze(1)).sum().item()

            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

            for i in range(out.size(0)):
                true_label = batch.y[i].item()
                pred_scores = out[i].cpu().numpy()
                rank = np.where(np.argsort(pred_scores)[
                                ::-1] == true_label)[0][0] + 1
                ranks.append(rank)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_rank = np.mean(ranks)
    total = len(test_data_list)
    accuracy = correct / total
    accuracy_top1 = correct_top1 / total
    accuracy_top3 = correct_top3 / total
    accuracy_top5 = correct_top5 / total
    print(f'Accuracy: {accuracy*100:.6f}% | Top-1: {accuracy_top1 * 100:.6f}% | Top-3: {accuracy_top3 *100:.6f}% | Top-5: {accuracy_top5 * 100:.6f}% | F1: {f1:.6f} | Mean Rank: {mean_rank:.6f}')


if __name__ == '__main__':
    graph_dir=r'D:\SX\wangan_code\data_prepare\data\data_random'
    model_dir='model_random'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    learn_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # learn_rates = [5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    num_layers_list = [1, 2, 3, 4]
    # learn_rates = [5e-3]
    # num_layers_list = [2]
    # graph_dir = 'data_2'
    epochs = 100
    for learn_rate in learn_rates:
        for num_layers in num_layers_list:
            print(f"LR={learn_rate}, K={num_layers}")
            train(graph_dir, epochs, learn_rate, num_layers, model_dir)

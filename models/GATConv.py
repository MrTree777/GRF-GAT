import torch
from torch_geometric.nn import GATConv, BatchNorm
from torch import nn
from torch_geometric.utils import dropout_adj
import networkx as nx
from collections import defaultdict, deque

"""
    args:
        in_channels：输入通道，比如节点分类中表示每个节点的特征数
        out_channels：输出通道，最后一层GCNConv的输出通道为节点类别数（节点分类）
        heads：多头注意力机制中的头数
        concat：如果为True，会将多个注意力机制的结果进行拼接，否则求平均
        negative_slope：LeakyRELU的参数
        add_self_loops：如果为False不再强制添加自环，默认为True
        bias：默认添加偏置
"""

class MH_GAT(torch.nn.Module):
    def __init__(self):
        super(MH_GAT, self).__init__()
        self.atrr = GGL()
        self.conv1 = GATConv(in_channels=272,out_channels=300,heads=7,concat=True,add_self_loops=True)
        self.bn1 = BatchNorm(2100)
        self.layer3 = nn.Sequential(
            nn.Linear(2100,256),
            nn.ReLU(inplace=True),
            nn.Dropout())

    def forward(self, x):
        # 先将输入特征 x 移动到 GPU 上

        edge_atrr, edge_index = self.atrr(x)  # 通过矩阵乘法得到边属性矩阵和边索引矩阵
        # edge_atrr , edge_index = full_connection_adj_matrix(x) #通过高斯核函数
        edge_atrr = edge_atrr.cuda()  # 移动到GPU
        edge_index = edge_index.cuda()  # 移动到GPU
        edge_index, edge_atrr = dropout_adj(edge_index, edge_atrr)

        G = nx.DiGraph()
        G.add_edges_from(edge_index.t().tolist())




        # 计算每个节点的入度数和出度数，并据此得到可学习的嵌入向量
        in_degrees = edge_index[1].bincount(minlength=x.size(0))
        out_degrees = edge_index[0].bincount(minlength=x.size(0))
        #以num_nodes为嵌入层的输入维度
        embedding_in_degree = torch.nn.Embedding(x.size(0),8).cuda()
        embedding_out_degree = torch.nn.Embedding(x.size(0),8).cuda()

        # 初始化参数为正值的均匀分布
        init_range = 1  # 你可以根据需要调整这个范围
        torch.nn.init.uniform_(embedding_in_degree.weight.data, 0, init_range)
        torch.nn.init.uniform_(embedding_out_degree.weight.data, 0, init_range)

        in_degrees_f = embedding_in_degree(in_degrees)
        out_degrees_f = embedding_out_degree(out_degrees)

        # shortest_paths = bfs_shortest_paths_0(edge_index)
        shortest_paths = bfs_shortest_paths_1(edge_index)

        edge_index, spa_pos= update_edge_index(edge_index, shortest_paths)
        edge_index = edge_index.cuda()
        spa_pos = spa_pos.cuda()

        x = torch.cat((x,in_degrees_f,out_degrees_f),dim=1)
        x = self.conv1(x = x,edge_index = edge_index,spa_pos = spa_pos)

        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x




class GGL(torch.nn.Module):
    '''
    Graph generation layer
    这是一个Graph generation layer（图形生成层）的实现，其作用是生成图形结构的连接方式，使得输入的数据能够构成图形结构
    '''
    def __init__(self,):
        super(GGL, self).__init__()
        self.layer = nn.Sequential(
            #输入是经过前面卷积和全连接层处理后的256维特征向量
            nn.Linear(256,10),
            nn.Sigmoid())
    #过一个线性层和sigmoid激活函数后输出10维向量作为每个节点的属性
    def forward(self, x):
        x = x.view(x.size(0), -1)#第一维(batch_size)保持不变，将剩下的元素展平为一维，并根据这一维的长度自动计算第二维的长度
        # atrr =x
        atrr = self.layer(x)    #这里将x输入到layer中，说明shape应该是256？输出的是10维
        values, edge_index = Gen_edge(atrr) #通过输出的属性，调用Gen_edge生成节点之间的连接边
        return values.view(-1), edge_index  #values展平成一个一维向量
        # values,edge_index = full_connection_adj_matrix(atrr)
        # return values,edge_index

def Gen_edge(atrr):
    atrr = atrr.cpu()   #将 Tensor 数据从 GPU 转到 CPU 上
    A = torch.mm(atrr, atrr.T)  #向量乘法，得到邻接矩阵
    maxval, maxind = A.max(axis=1)  #得到每行得最大值，及其对应得下标
    A_norm = A / maxval             #将每行最大值用于归一化邻接矩阵A
    # k = A.shape[0]                  #batchsize
    k=20
    values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False)
    edge_index = torch.tensor([[],[]],dtype=torch.long) #这是一个二维的 PyTorch 张量（tensor），其中包含两个空列表，数据类型为long tensor([], size=(2, 0), dtype=torch.int64)


    for i in range(indices.shape[0]):   #遍历从0-9
        index_1 = torch.zeros(indices.shape[1],dtype=torch.long) + i    #10维的一个tensor
        index_2 = indices[i]                                            #10维的一个tensor
        sub_index = torch.stack([index_1,index_2])  #将张量按照指定维度合并成一个新的张量，默认dim=0 按行合并 每次得到一个2X10的矩阵
        edge_index = torch.cat([edge_index,sub_index],axis=1)   #将上面的矩阵按列cat for循环结束后得到的是一个2 X 100的矩阵   边索引矩阵 上面一行是源节点 下面一行表示目标节点
    return values, edge_index       #values可以看成边权重矩阵吗？



def rbf_kernel(X, Y, gamma):
    """
    计算RBF核函数
    """
    XX = torch.matmul(X, X.t())
    XY = torch.matmul(X, Y.t())
    YY = torch.matmul(Y, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    rbf_kernel = torch.exp(-gamma * (
            -2 * XY + X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0)
    ))

    return rbf_kernel

def full_connection_adj_matrix(atrr, gamma=1.0, k=20):
    atrr = atrr.cpu()   # 将 Tensor 数据从 GPU 转到 CPU 上
    A = rbf_kernel(atrr, atrr, gamma)  # 使用高斯核函数计算相似度矩阵

    # 对邻接矩阵A进行处理，取top-k，得到边索引和边权重
    values, indices = A.topk(k, dim=1, largest=True, sorted=False)
    edge_index = torch.tensor([[],[]], dtype=torch.long)
    for i in range(indices.shape[0]):
        index_1 = torch.zeros(indices.shape[1], dtype=torch.long) + i
        index_2 = indices[i]
        sub_index = torch.stack([index_1, index_2])
        edge_index = torch.cat([edge_index, sub_index], axis=1)

    return values.view(-1), edge_index


def edge_index_to_shortest_paths(edge_index):
    # 将边索引转换为NetworkX有向图
    G = nx.DiGraph()
    # num_nodes = max(edge_index[0])+1

    all_nodes = torch.cat((edge_index[0], edge_index[1]), dim=0)
    unique_nodes = torch.unique(all_nodes)
    # 计算节点数量
    num_nodes = int(torch.max(unique_nodes) + 1)


    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)

    # 使用-1初始化最短路径矩阵
    shortest_paths = -torch.ones((num_nodes, num_nodes), dtype=torch.float)

    # 计算每对节点之间的最短路径
    for node in range(num_nodes):
        # 计算从节点到所有其他节点的最短路径
        dist_dict = nx.single_source_shortest_path_length(G, node)
        for target, dist in dist_dict.items():
            # 对于有自环的节点，将最短路径设为0
            if node == target:
                shortest_paths[node, target] = 0
            # 对于没有路径的节点，将最短路径设为-1
            elif dist == float('inf'):
                shortest_paths[node, target] = -1
            else:
                shortest_paths[node, target] = dist

    return shortest_paths


def update_edge_index(edge_index, shortest_paths):
    # 获取图中节点的数量
    num_nodes = shortest_paths.size(0)

    # 将最短路径矩阵转换为布尔类型，其中True表示节点之间有最短路径，False表示没有路径或自环
    shortest_paths_bool = shortest_paths != -1

    # 初始化存储更新后的边索引和边权重的列表
    updated_edge_index_list = []
    spa_pos_list = []

    # 遍历最短路径矩阵，提取边索引和边权重
    for i in range(num_nodes):
        for j in range(num_nodes):
             # if shortest_paths[i, j] >= 0 and shortest_paths[i, j] <= 1:
             if shortest_paths_bool[i, j]:
                updated_edge_index_list.append(torch.tensor([i, j]))
                spa_pos_list.append(shortest_paths[i, j])

    updated_edge_index = torch.stack(updated_edge_index_list, dim=1)
    spa_pos = torch.stack(spa_pos_list)

    return updated_edge_index, spa_pos


"""自环设置为0"""
def bfs_shortest_paths_0(edge_index):
    num_nodes = max(edge_index[0]) + 1
    graph = defaultdict(list)

    # Convert edge index matrix to adjacency list
    for i in range(len(edge_index[0])):
        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()
        graph[src].append(dst)

    shortest_paths = torch.full((num_nodes, num_nodes), -1, dtype=torch.float)

    for start_node in range(num_nodes):
        visited = [False for _ in range(num_nodes)]  # 记录已访问的节点
        queue = deque([(start_node, 0)])  # 初始化队列，用起始节点和距离0
        visited[start_node] = True

        while queue:
            curr_node, dist = queue.popleft()

            # 处理自环的情况
            if curr_node == start_node:
                shortest_paths[start_node][curr_node] = 0
            else:
                # 更新当前节点和起始节点之间的最短路径
                shortest_paths[start_node][curr_node] = dist

            # 探索当前节点的邻居节点
            for neighbor in graph[curr_node]:
                if not visited[neighbor]:
                    queue.append((neighbor, dist + 1))
                    visited[neighbor] = True

    return shortest_paths
"""自环设置为1"""
def bfs_shortest_paths_1(edge_index):
    num_nodes = max(edge_index[0]) + 1
    graph = defaultdict(list)

    # Convert edge index matrix to adjacency list
    for i in range(len(edge_index[0])):
        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()
        graph[src].append(dst)

    shortest_paths = torch.full((num_nodes, num_nodes), -1, dtype=torch.float)

    for start_node in range(num_nodes):
        visited = [False for _ in range(num_nodes)]  # 记录已访问的节点
        queue = deque([(start_node, 0)])  # 初始化队列，用起始节点和距离0
        visited[start_node] = True

        while queue:
            curr_node, dist = queue.popleft()

            # 处理自环的情况
            if curr_node == start_node:
                shortest_paths[start_node][curr_node] = 1  # 自环的最短路径长度设置为1
            elif shortest_paths[start_node][curr_node] == -1:
                shortest_paths[start_node][curr_node] = dist

            if dist < start_node:  # 限制路径长度，如一阶邻居的最短路径长度为2
                # 探索当前节点的邻居节点
                for neighbor in graph[curr_node]:
                    if not visited[neighbor]:
                        queue.append((neighbor, dist + 1))
                        visited[neighbor] = True

    return shortest_paths

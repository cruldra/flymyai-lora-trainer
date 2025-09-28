import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 联邦学习：迈向隐私保护机器学习的关键一步

    学习以数据隐私为主要关注点的真实世界ML模型开发——实用指南。

    在许多实际的机器学习(ML)项目中，将数据整合到中央位置是一种常见做法。

    ![传统ML方法](https://www.dailydoseofds.com/content/images/2023/11/image-40.png)

    随后，机器学习工程师利用这些集中化的数据进行：

    - 分析
    - 进行特征工程
    - 最终进行模型训练、验证、扩展、部署和持续的生产监控

    ![ML流程](https://www.dailydoseofds.com/content/images/2023/11/image-41.png)

    这种传统方法被广泛接受并用于开发ML模型。

    然而，与这种传统方法相关的一个显著挑战是，它要求数据在任何后续处理发生之前必须物理集中化。

    让我们详细了解这方面的问题！

    ---

    ## 传统ML建模的问题

    考虑我们的应用程序拥有数百万用户基础。显然，要处理的数据量可能极其庞大。

    这些数据很有价值，因为现代设备可以访问大量适合机器学习模型的数据。

    ![设备数据](https://www.dailydoseofds.com/content/images/2023/11/image-44.png)

    这些数据可以显著改善设备上的用户体验。

    例如：

    - 如果是文本数据，那么语言模型可以改善语音识别和文本输入
    - 如果是图像数据，那么许多下游图像模型可以得到改善，等等

    然而，传统的机器学习方法，即将所有数据聚合在中央存储库中，在这种情况下会带来许多挑战。

    更具体地说，在这种方法中，将数据从个人用户设备传输到中央位置既耗费带宽又耗费时间，这会阻止用户参与。

    ![数据传输问题](https://www.dailydoseofds.com/content/images/2023/11/image-45.png)

    即使用户被激励贡献数据，在用户设备和中央服务器上都有数据的冗余性可能在逻辑上是不可行的，因为我们可能要处理的数据量巨大。

    此外，数据通常包含个人信息，如照片、私人文本和语音笔记。

    ![隐私数据](https://www.dailydoseofds.com/content/images/2023/11/image-47.png)

    要求用户上传如此敏感的数据不仅危及隐私，还引发法律问题。将此类数据存储在集中式数据库中变得有问题，引入了可行性问题和隐私违规。

    这在将此数据存储在集中式数据库中造成了问题。简单地说，这既可能不可行，又会引发许多隐私违规。

    将大量数据移动到中央服务器在用户带宽和时间方面可能成本高昂。

    但数据对我们仍然有价值，不是吗？我们想以某种方式利用它。

    **联邦学习**是一种令人难以置信的机器学习模型训练技术，它最小化数据传输，使其适用于低带宽和高延迟环境。

    ![联邦学习概念](https://www.dailydoseofds.com/content/images/2023/11/Screen-Recording-2023-11-13-at-1.46.14-PM.gif)

    让我们来了解一下！
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 联邦学习如何解决这些问题

    正式地说，联邦学习代表了一种去中心化的机器学习方法，其中训练数据保留在个人设备(如智能手机)上的本地位置。

    不是将数据传输到中央服务器，而是将模型分发到设备上，在本地进行训练，只有结果模型更新被收集并发送回服务器。

    ![联邦学习流程](https://www.dailydoseofds.com/content/images/2023/11/Screen-Recording-2023-11-13-at-5.09.42-PM.gif)

    本质上，这种方法涉及将训练数据留在个人设备上，同时通过聚合本地计算的梯度更新来学习共享模型。

    💡

    术语"联邦学习"源于这样一个概念：参与设备(称为客户端)的松散**联邦**与中央服务器协作解决学习任务。

    其主要优点之一在于通过消除对集中式数据收集的所有依赖来增强隐私和安全性。

    这是因为每个客户端都拥有一个本地训练数据集，该数据集完全保留在设备上，永远不会上传到服务器。

    相反，客户端计算对服务器维护的全局模型的更新，只传输必要的"模型更新"。

    ![模型更新](https://www.dailydoseofds.com/content/images/2023/11/image-48.png)

    因此，整个模型更新过程发生在客户端，通过将模型训练与直接访问原始训练数据的必要性解耦，提供了关键优势。

    虽然需要对协调服务器有一定程度的信任，但联邦学习有效地解决了与传统集中式机器学习模型训练方法相关的主要问题。

    通过促进设备上训练并最小化对大量数据传输的需求，联邦学习为传统模型训练范式中固有的挑战提供了实用解决方案。

    使用联邦学习的主要动机是：

    - **隐私**：
        - 保护用户数据是首要任务，特别是因为最近越来越多的用户开始关心他们的隐私
        - 集中式数据存储库带来固有的隐私风险，而联邦学习通过允许数据完全驻留在用户设备上来缓解这些问题，最小化暴露

    - **带宽和延迟**：
        - 如前所述，将大量数据传输到中央服务器的资源密集型过程既耗时又耗带宽
        - 联邦学习战略性地最小化数据传输，在低带宽和高延迟环境中特别有利

    - **数据所有权**：
        - 用户在联邦学习框架内保持对其数据的控制和所有权
        - 这不仅解决了与数据所有权相关的问题，还确保了数据权利的保护，提供了以用户为中心的机器学习方法

    - **可扩展性**：
        - 联邦学习展现出与设备数量增加无缝对齐的自然可扩展性
        - 这种固有的可扩展性使其非常适合大规模应用，涵盖移动设备、IoT设备和边缘计算场景

    本质上，联邦学习代表了一种范式转变，将我们的模型带到数据所在的地方，而不是将数据移动到模型所在的位置的传统方法。

    这种传统模型训练过程的颠倒强调了联邦学习在当代数据驱动应用中的适应性和效率。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 联邦学习系统如何提供隐私保护？

    当然，在这一点上，可能会出现这样的论点：在将数据上传到中央服务器之前对其进行匿名化可以解决隐私问题。

    简单地说，匿名化意味着从数据集中删除所有个人可识别信息(PII)。

    这通常涉及替换或加密特定数据元素，以防止识别与信息相关的个人。

    然而，与普遍看法相反，即使处理匿名化数据也可能引入隐私问题。

    考虑一个持卡人数据库的场景——一个高度敏感的数据集。

    虽然掩盖卡号是一种常见做法，但处理所需的其他详细信息(如持卡人地址)可能仍然存在。

    因此，匿名化数据集并不总是保证消除隐私问题。

    另一方面，联邦学习最小化了向集中位置传输数据特定信息。如上所述，传输的信息是最小的，通常包含显著更少的原始数据。

    在这种范式中，只有**模型更新**被发送到中央服务器，值得注意的是，服务器端的聚合算法不需要了解这些更新的来源。因此，来源信息可以完全被忽略。

    ![匿名性保证](https://www.dailydoseofds.com/content/images/2023/11/image-49.png)

    这种对来源信息的不依赖通过确保本地生成的模型更新可以在不透露任何可能危及用户隐私的其他详细信息的情况下传输，从而保证了真正的匿名性。

    这创造了一个互利的场景：

    - 用户满意，因为他们的体验由高质量的ML模型驱动，而不会危及他们的数据
    - 同时，团队通过成功解决各种挑战而受益，包括：

    1. **隐私问题**：联邦学习有效地避开了与传统集中式方法相关的隐私问题
    2. **降低模型训练成本**：该方法有助于减轻与集中式模型训练相关的成本
    3. **最小化数据维护成本**：联邦学习显著减少了数据维护成本的负担
    4. **大数据集训练**：团队可以在广泛的数据集上训练模型，而无需集中存储
    5. **更好的用户体验**：尽管没有集中式数据存储，仍可以开发高质量的ML模型

    本质上，联邦学习为每个人提供了双赢的解决方案。

    ### 联邦学习的额外好处

    #### 更多数据暴露

    在联邦学习中，用于模型训练的数据范围超出了集中式数据工程可能收集和管理的范围。

    ![数据多样性](https://www.dailydoseofds.com/content/images/2023/11/image-50.png)

    通过利用驻留在个人用户设备上的全部数据谱，联邦学习使模型能够从多样化和丰富的数据集中学习。

    这种多样性增强了模型的鲁棒性，使它们更能代表真实世界的场景。

    #### 互利

    联邦学习不仅通过协作训练改进模型，还直接为用户带来好处。

    当用户的设备参与模型训练时，它会基于集体知识接收更新，从而增强用户体验。

    例如，在个性化推荐系统中，用户受益于在更大用户群偏好上训练的模型，从而获得更准确和定制的推荐。

    #### 有限的计算要求

    与需要大量计算资源在中央服务器上进行数据处理和模型训练的传统集中式方法不同，联邦学习将大部分计算重新分配给用户设备。

    ![计算分布](https://www.dailydoseofds.com/content/images/2023/11/image-51.png)

    这种转变带来了几个优势：

    - **减少服务器负载**：中央服务器需要更少的计算能力，因为它们不再需要处理和训练大量数据
    - **更低延迟**：用户体验更低的延迟，因为数据不需要传输到远程服务器进行处理，从而改善整体用户体验
    - **能源效率**：用户设备上的本地计算可能更节能
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 何时适合使用联邦学习？

    在理解联邦学习的关键策略之前，必须理解联邦学习的适用性不是一刀切的命题。

    与其在任何地方都采用它，理解联邦学习是最佳方法的特定情况是至关重要的。

    ![适用场景](https://www.dailydoseofds.com/content/images/2023/11/image-53.png)

    这是因为如果你理解了这些特定类型的情况并且某天遇到它们，你会立即知道联邦学习是这里的出路。

    根据我的经验，联邦学习的理想问题具有以下特性：

    #### 代理数据不合适

    **场景**：边缘设备(如手机)上可用的训练数据相对于公共数据集或中央服务器的代理数据具有独特优势。

    **理由**：优先利用直接来自边缘设备的数据，因为它提供了目标环境的更真实表示。在代理数据无法捕获真实世界场景复杂性的情况下，联邦学习成为更有前景的解决方案。

    #### 理想数据具有隐私敏感性

    **场景**：理想的训练数据表现出隐私敏感或相对于模型容量显著大的特征。

    **理由**：当数据包含敏感信息或数据量巨大时，传统的集中式方法变得不切实际。联邦学习允许在不危及隐私或需要大规模数据传输的情况下利用这些数据。

    #### 设备上的数据是动态的

    **场景**：用户设备上的数据不断变化和更新，反映了用户行为和偏好的演变。

    **理由**：联邦学习可以适应这种动态性质，允许模型持续从新数据中学习，而无需不断将更新的数据集传输到中央位置。

    #### 网络连接不稳定

    **场景**：设备可能具有间歇性或有限的网络连接。

    **理由**：联邦学习的设计可以处理这种情况，因为它只需要偶尔的连接来发送模型更新，而不是持续的数据流。

    #### 法规合规要求

    **场景**：存在严格的数据保护法规，如GDPR、HIPAA等。

    **理由**：联邦学习通过确保敏感数据永远不离开其原始位置来帮助满足这些法规要求，同时仍然允许从数据中获得见解。

    ---

    ## 联邦学习的核心算法

    现在让我们深入了解联邦学习中使用的两种最流行的算法。

    ### 1. 联邦平均算法(FedAvg)

    联邦平均算法是联邦学习中最基础和广泛使用的算法之一。

    #### 算法步骤：

    1. **初始化**：服务器初始化全局模型参数
    2. **客户端选择**：服务器选择一部分客户端参与当前轮次的训练
    3. **模型分发**：服务器将当前全局模型发送给选定的客户端
    4. **本地训练**：每个客户端在其本地数据上训练模型几个epoch
    5. **模型上传**：客户端将更新后的模型参数发送回服务器
    6. **聚合**：服务器对所有客户端模型进行加权平均
    7. **重复**：重复步骤2-6直到收敛

    #### 数学表示：

    ```
    全局模型更新：w_t+1 = Σ(n_k/n) * w_k_t+1

    其中：
    - w_t+1 是第t+1轮的全局模型参数
    - w_k_t+1 是客户端k在第t+1轮的本地模型参数
    - n_k 是客户端k的数据样本数
    - n 是所有客户端的总样本数
    ```

    #### 优点：
    - 简单易实现
    - 通信效率高
    - 在IID数据上表现良好

    #### 缺点：
    - 在非IID数据上性能下降
    - 可能存在客户端漂移问题

    ### 2. 联邦近端算法(FedProx)

    FedProx是FedAvg的改进版本，专门设计来处理非IID数据和系统异构性。

    #### 关键改进：

    1. **近端项**：在本地目标函数中添加近端项，防止本地模型偏离全局模型太远
    2. **部分参与**：允许客户端进行不完整的本地更新

    #### 目标函数：

    ```
    本地目标函数：F_k(w) + (μ/2)||w - w_t||²

    其中：
    - F_k(w) 是客户端k的原始损失函数
    - μ 是近端项系数
    - w_t 是当前全局模型参数
    ```

    #### 优点：
    - 更好地处理非IID数据
    - 对系统异构性更鲁棒
    - 理论收敛保证更强

    #### 缺点：
    - 需要调整额外的超参数μ
    - 计算开销略高
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PyTorch中的联邦学习实现

    让我们通过一个实际的例子来看看如何在PyTorch中实现联邦学习。

    ### 基础设置

    首先，我们需要设置基本的环境和数据：

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成示例数据
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_classes=2,
        random_state=42
    )

    # 将数据转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    ```

    ### 定义神经网络模型

    ```python
    class SimpleNN(nn.Module):
        def __init__(self, input_size=20, hidden_size=64, num_classes=2):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # 创建全局模型
    global_model = SimpleNN()
    ```

    ### 数据分布模拟

    为了模拟真实的联邦学习场景，我们需要将数据分布到不同的客户端：

    ```python
    def create_client_data(X, y, num_clients=5, iid=True):
        \"\"\"
        将数据分布到不同的客户端

        参数:
        - X, y: 输入数据和标签
        - num_clients: 客户端数量
        - iid: 是否创建IID分布
        \"\"\"
        client_data = []

        if iid:
            # IID分布：随机分配数据
            indices = np.random.permutation(len(X))
            split_indices = np.array_split(indices, num_clients)

            for indices in split_indices:
                client_X = X[indices]
                client_y = y[indices]
                dataset = TensorDataset(client_X, client_y)
                client_data.append(DataLoader(dataset, batch_size=32, shuffle=True))
        else:
            # 非IID分布：基于标签分配数据
            class_indices = {i: np.where(y == i)[0] for i in range(2)}

            for i in range(num_clients):
                # 每个客户端主要获得一个类别的数据
                primary_class = i % 2
                secondary_class = 1 - primary_class

                # 80%来自主要类别，20%来自次要类别
                primary_size = int(0.8 * len(class_indices[primary_class]) / (num_clients // 2))
                secondary_size = int(0.2 * len(class_indices[secondary_class]) / (num_clients // 2))

                primary_indices = np.random.choice(
                    class_indices[primary_class],
                    size=min(primary_size, len(class_indices[primary_class])),
                    replace=False
                )
                secondary_indices = np.random.choice(
                    class_indices[secondary_class],
                    size=min(secondary_size, len(class_indices[secondary_class])),
                    replace=False
                )

                client_indices = np.concatenate([primary_indices, secondary_indices])
                client_X = X[client_indices]
                client_y = y[client_indices]

                dataset = TensorDataset(client_X, client_y)
                client_data.append(DataLoader(dataset, batch_size=32, shuffle=True))

        return client_data

    # 创建客户端数据
    iid_client_data = create_client_data(X_tensor, y_tensor, num_clients=5, iid=True)
    non_iid_client_data = create_client_data(X_tensor, y_tensor, num_clients=5, iid=False)
    ```

    ### 联邦平均算法实现

    ```python
    def train_client_model(model, dataloader, epochs=5, lr=0.01):
        \"\"\"
        在客户端训练模型
        \"\"\"
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def federated_averaging(global_model, client_data, num_rounds=10, client_epochs=5):
        \"\"\"
        联邦平均算法实现
        \"\"\"
        global_weights = global_model.state_dict()

        for round_num in range(num_rounds):
            print(f"联邦学习轮次 {round_num + 1}/{num_rounds}")

            # 存储客户端权重
            client_weights = []

            # 每个客户端训练
            for client_id, dataloader in enumerate(client_data):
                # 创建客户端模型副本
                client_model = SimpleNN()
                client_model.load_state_dict(global_weights)

                # 训练客户端模型
                updated_weights = train_client_model(
                    client_model, dataloader, epochs=client_epochs
                )
                client_weights.append(updated_weights)

            # 聚合权重（简单平均）
            aggregated_weights = {}
            for key in global_weights.keys():
                aggregated_weights[key] = torch.stack([
                    client_weights[i][key] for i in range(len(client_weights))
                ]).mean(dim=0)

            # 更新全局模型
            global_weights = aggregated_weights
            global_model.load_state_dict(global_weights)

        return global_model
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 实验结果对比

    让我们比较不同场景下的性能：

    ```python
    def evaluate_model(model, X_test, y_test):
        \"\"\"
        评估模型性能
        \"\"\"
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
        return accuracy

    # 准备测试数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    # 场景1：IID数据的联邦学习
    print("=== IID数据联邦学习 ===")
    iid_model = SimpleNN()
    iid_trained_model = federated_averaging(
        iid_model, iid_client_data, num_rounds=10
    )
    iid_accuracy = evaluate_model(iid_trained_model, X_test, y_test)
    print(f"IID联邦学习准确率: {iid_accuracy:.4f}")

    # 场景2：非IID数据的联邦学习
    print("\\n=== 非IID数据联邦学习 ===")
    non_iid_model = SimpleNN()
    non_iid_trained_model = federated_averaging(
        non_iid_model, non_iid_client_data, num_rounds=10
    )
    non_iid_accuracy = evaluate_model(non_iid_trained_model, X_test, y_test)
    print(f"非IID联邦学习准确率: {non_iid_accuracy:.4f}")

    # 场景3：集中式训练（基准）
    print("\\n=== 集中式训练（基准） ===")
    centralized_model = SimpleNN()
    centralized_dataset = TensorDataset(X_train, y_train)
    centralized_dataloader = DataLoader(centralized_dataset, batch_size=32, shuffle=True)

    # 训练集中式模型
    centralized_model.train()
    optimizer = optim.SGD(centralized_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):  # 更多epoch以确保收敛
        for batch_x, batch_y in centralized_dataloader:
            optimizer.zero_grad()
            outputs = centralized_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    centralized_accuracy = evaluate_model(centralized_model, X_test, y_test)
    print(f"集中式训练准确率: {centralized_accuracy:.4f}")
    ```

    ### 结果分析

    典型的实验结果可能如下：

    ```
    === IID数据联邦学习 ===
    联邦学习轮次 1/10
    联邦学习轮次 2/10
    ...
    IID联邦学习准确率: 0.8750

    === 非IID数据联邦学习 ===
    联邦学习轮次 1/10
    联邦学习轮次 2/10
    ...
    非IID联邦学习准确率: 0.7850

    === 集中式训练（基准） ===
    集中式训练准确率: 0.9100
    ```

    从结果可以看出：

    1. **集中式训练**表现最好，因为它可以访问所有数据
    2. **IID联邦学习**性能接近集中式训练
    3. **非IID联邦学习**性能显著下降，这是数据异构性造成的

    ### 非IID数据问题的可视化

    让我们可视化不同客户端的数据分布：

    ```python
    import matplotlib.pyplot as plt

    def visualize_data_distribution(client_data, title):
        \"\"\"
        可视化客户端数据分布
        \"\"\"
        fig, axes = plt.subplots(1, len(client_data), figsize=(15, 3))
        fig.suptitle(title)

        for i, dataloader in enumerate(client_data):
            labels = []
            for _, batch_y in dataloader:
                labels.extend(batch_y.numpy())

            unique, counts = np.unique(labels, return_counts=True)
            axes[i].bar(unique, counts)
            axes[i].set_title(f'客户端 {i+1}')
            axes[i].set_xlabel('类别')
            axes[i].set_ylabel('样本数量')

        plt.tight_layout()
        plt.show()

    # 可视化IID和非IID分布
    visualize_data_distribution(iid_client_data, "IID数据分布")
    visualize_data_distribution(non_iid_client_data, "非IID数据分布")
    ```

    这个可视化清楚地显示了：
    - **IID分布**：每个客户端都有相似的类别分布
    - **非IID分布**：每个客户端的数据严重偏向某个类别

    这种数据异构性是联邦学习面临的主要挑战之一。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 非IID数据问题的解决方案

    ### 问题分析

    在真实世界中，联邦学习面临的最大挑战之一是数据的非独立同分布(非IID)特性。

    考虑以下现实场景：
    - 一个客户端可能是宠物爱好者——在他们的设备上积累了大量的猫和狗图像
    - 另一个人可能是F1赛车迷，收集的图像专门以赛车为特色
    - 你的设备也是一个客户端，你的设备中最多的是什么类型的图像？

    **关键点是，隐私敏感数据集几乎总是带有个人喜好和信念的偏见，这些总是因人而异。**

    因此，IID数据集的假设成为一个显著的过度简化，因为客户端之间的数据分布可能表现出显著的异构性。

    这种偏好差异导致客户端之间的非IID数据分布。每个用户的数据不仅独立于其他用户，而且明显不同，反映了他们的个人兴趣和偏好。

    ### 传统联邦学习的局限性

    让我们回顾一下训练循环：

    ```python
    # 传统联邦学习步骤：
    # 1. 每个客户端都收到相同的模型
    # 2. 他们在本地数据上训练模型
    # 3. 他们返回训练后模型的权重
    # 4. 在全局模型中聚合权重
    # 5. 重复
    ```

    当数据是非IID时，这种方法会导致：
    - **客户端漂移**：每个客户端的模型朝着其本地数据分布的方向优化
    - **聚合冲突**：不同客户端的更新可能相互冲突
    - **性能下降**：全局模型性能显著降低

    ### 解决方案：引入公共数据集

    为了缓解联邦学习中的非IID数据挑战，我们经常引入一个小而均匀的数据集，用于进一步训练步骤3中所有客户端返回的模型。

    当模型在非IID数据集上训练时，它可能会偏向于具有更多样本或更具挑战性示例的客户端的数据分布。

    在小而均匀的数据集上进行微调有助于模型适应整体数据的一般特征，同时也了解本地数据特征。

    鉴于这个公共数据集相对较小，策划它并不是一个大挑战。

    **因此，更新后的步骤变为：**

    1. 每个客户端都收到相同的模型
    2. 他们在本地数据上训练模型
    3. 他们返回训练后模型的权重
    4. 在小数据集上进一步调整客户端模型
    5. 聚合本地模型的权重
    6. 重复

    ### 改进的联邦学习实现

    ```python
    def federated_averaging_with_common_data(global_model, client_data, common_dataloader,
                                            num_rounds=10, client_epochs=5, common_epochs=2):
        \"\"\"
        带有公共数据集的改进联邦平均算法
        \"\"\"
        global_weights = global_model.state_dict()

        for round_num in range(num_rounds):
            print(f"改进联邦学习轮次 {round_num + 1}/{num_rounds}")

            # 存储客户端权重
            client_weights = []

            # 每个客户端训练
            for client_id, dataloader in enumerate(client_data):
                # 创建客户端模型副本
                client_model = SimpleNN()
                client_model.load_state_dict(global_weights)

                # 步骤1-3：在本地数据上训练
                updated_weights = train_client_model(
                    client_model, dataloader, epochs=client_epochs
                )

                # 步骤4：在公共数据上微调
                client_model.load_state_dict(updated_weights)
                fine_tuned_weights = train_client_model(
                    client_model, common_dataloader, epochs=common_epochs
                )

                client_weights.append(fine_tuned_weights)

            # 步骤5：聚合权重
            aggregated_weights = {}
            for key in global_weights.keys():
                aggregated_weights[key] = torch.stack([
                    client_weights[i][key] for i in range(len(client_weights))
                ]).mean(dim=0)

            # 更新全局模型
            global_weights = aggregated_weights
            global_model.load_state_dict(global_weights)

        return global_model

    # 创建公共数据集（小而均匀）
    def create_common_dataset(X, y, size=200):
        \"\"\"
        创建小而均匀的公共数据集
        \"\"\"
        # 确保每个类别都有相等的样本数
        class_indices = {i: np.where(y == i)[0] for i in range(2)}
        samples_per_class = size // 2

        common_indices = []
        for class_label in range(2):
            selected_indices = np.random.choice(
                class_indices[class_label],
                size=samples_per_class,
                replace=False
            )
            common_indices.extend(selected_indices)

        common_indices = np.array(common_indices)
        common_X = X[common_indices]
        common_y = y[common_indices]

        dataset = TensorDataset(common_X, common_y)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建公共数据集
    common_dataloader = create_common_dataset(X_tensor, y_tensor, size=200)

    # 使用改进算法训练
    print("=== 改进的非IID联邦学习 ===")
    improved_model = SimpleNN()
    improved_trained_model = federated_averaging_with_common_data(
        improved_model, non_iid_client_data, common_dataloader, num_rounds=10
    )
    improved_accuracy = evaluate_model(improved_trained_model, X_test, y_test)
    print(f"改进联邦学习准确率: {improved_accuracy:.4f}")
    ```

    ### 性能对比

    典型的改进结果：

    ```
    === 改进的非IID联邦学习 ===
    改进联邦学习轮次 1/10
    改进联邦学习轮次 2/10
    ...
    改进联邦学习准确率: 0.8650
    ```

    **性能对比总结：**

    | 方法 | 准确率 | 改进幅度 |
    |------|--------|----------|
    | 集中式训练 | 91.0% | 基准 |
    | IID联邦学习 | 87.5% | -3.5% |
    | 非IID联邦学习 | 78.5% | -12.5% |
    | 改进非IID联邦学习 | 86.5% | +8.0% |

    如图所示，在小数据集上进一步训练模型相比标准非IID客户端训练，准确率提高了约**8-10%**。

    ### 其他改进策略

    除了公共数据集方法，还有其他策略来处理非IID数据：

    #### 1. 个性化联邦学习
    ```python
    # 为每个客户端维护个性化层
    def personalized_federated_learning(global_model, client_data):
        # 全局层共享，个性化层本地保留
        pass
    ```

    #### 2. 联邦蒸馏
    ```python
    # 使用知识蒸馏技术
    def federated_distillation(teacher_models, student_model):
        # 从多个教师模型中蒸馏知识
        pass
    ```

    #### 3. 数据增强
    ```python
    # 在客户端进行数据增强以增加多样性
    def augment_client_data(dataloader):
        # 应用数据增强技术
        pass
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    通过这次对联邦学习的实用深入研究，我们学到了很多：

    ### 📚 **关键学习要点**

    - **传统机器学习训练范式的挑战**
    - **什么是联邦学习？**
    - **联邦学习如何解决上述挑战**
    - **何时使用联邦学习？**
    - **联邦学习的两种最流行算法**
    - **在PyTorch中实现它们**
    - **偏离IID数据集并测量性能**
    - **如何缓解非IID数据集的问题**

    ### 🎯 **核心洞察**

    如果你仔细观察，联邦学习背后的想法非常有趣和聪明。

    **不是将数据收集到一个地方，我们将模型发送到数据所在的位置，从它们那里收集更新，并将它们聚合到一个地方。**

    这很聪明，不是吗？

    这种方法不仅通过最小化数据传输来保护用户隐私，而且在低带宽和高延迟环境中也被证明是有效的。

    ### 🔍 **实际应用价值**

    #### **隐私保护**
    - 用户数据永远不离开设备
    - 只传输模型更新，不传输原始数据
    - 符合GDPR、HIPAA等隐私法规

    #### **资源效率**
    - 减少中央服务器的计算负担
    - 利用边缘设备的计算能力
    - 降低网络带宽需求

    #### **可扩展性**
    - 自然适应设备数量的增长
    - 支持大规模分布式学习
    - 适用于IoT和移动设备场景

    ### ⚠️ **挑战与限制**

    #### **技术挑战**
    - **非IID数据**：客户端数据分布不均匀
    - **系统异构性**：设备计算能力差异
    - **通信效率**：网络带宽和延迟限制
    - **安全性**：模型更新可能泄露信息

    #### **实际部署挑战**
    - **客户端选择**：如何选择参与训练的客户端
    - **激励机制**：如何激励用户参与
    - **模型聚合**：如何有效聚合异构更新
    - **故障处理**：如何处理客户端掉线

    ### 🚀 **未来发展方向**

    #### **算法改进**
    - 更好的非IID数据处理方法
    - 自适应聚合算法
    - 个性化联邦学习
    - 异步联邦学习

    #### **系统优化**
    - 更高效的通信协议
    - 动态客户端选择策略
    - 边缘计算集成
    - 跨平台兼容性

    #### **安全增强**
    - 差分隐私集成
    - 安全多方计算
    - 同态加密应用
    - 拜占庭容错机制

    ### 💡 **实践建议**

    #### **何时选择联邦学习**
    1. 数据具有隐私敏感性
    2. 数据分布在多个设备/组织
    3. 数据传输成本高昂
    4. 需要满足法规合规要求
    5. 希望利用边缘计算能力

    #### **实施最佳实践**
    1. **数据质量控制**：确保客户端数据质量
    2. **模型设计**：选择适合联邦学习的模型架构
    3. **通信优化**：使用模型压缩和量化技术
    4. **安全考虑**：实施适当的隐私保护机制
    5. **性能监控**：建立有效的模型性能监控体系

    ### 🎪 **总结思考**

    虽然本文提供的联邦学习示例专注于说明工作流程和概念，但现实世界的应用将必须处理更多的设计和扩展挑战。

    联邦学习代表了机器学习领域的一个重要范式转变，它平衡了模型性能、数据隐私和系统效率之间的关系。

    随着隐私保护意识的增强和边缘计算的发展，联邦学习将在未来的AI系统中发挥越来越重要的作用。

    **记住**：联邦学习不是万能的解决方案，而是在特定场景下的优秀选择。理解其适用性、优势和局限性，才能在实际项目中做出明智的技术决策。

    ---

    💡 **下一步学习**：我们将在即将到来的文章中涵盖实际联邦学习的许多其他细节和挑战，包括更多设计和扩展挑战的深入讨论。
    """
    )
    return


if __name__ == "__main__":
    app.run()

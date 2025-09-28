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
    # 完整的MLOps蓝图：ML系统中的可重现性和版本控制—第A部分（含实现）

    ## 回顾

    在深入MLOps和LLMOps速成课程的第3部分之前，让我们简要回顾一下在课程的前一部分中涵盖的内容。

    在第2部分中，我们深入研究了ML系统生命周期，将不同阶段分解为详细解释。

    ![ML系统生命周期](https://www.dailydoseofds.com/content/images/2025/08/image-27.png)

    我们首先了解了ML系统生命周期的第一阶段：数据管道。它涉及摄取、存储、ETL、标注/注释和数据版本控制等关键概念。

    ![数据管道](https://www.dailydoseofds.com/content/images/2025/08/image-28.png)

    接下来，我们查看了模型训练和实验阶段，在那里我们讨论了与实验跟踪、模型选择、验证、训练管道、资源管理和超参数配置相关的关键想法。

    ![模型训练和实验](https://www.dailydoseofds.com/content/images/2025/08/image-29.png)

    进一步，我们探索了部署和推理阶段。我们通过理解模型打包、推理和部署方法、测试、模型注册表和CI/CD等内容深入研究。

    ![部署和推理](https://www.dailydoseofds.com/content/images/2025/08/image-30.png)

    在部署和推理之后，我们继续了解监控和可观察性。在那里，我们查看了运营监控、漂移和性能监控。

    ![监控和可观察性](https://www.dailydoseofds.com/content/images/2025/08/image-31.png)

    最后，我们通过一个快速的实践模拟，研究了如何序列化模型、将它们转换为FastAPI服务、测试它们，并为可重现性和部署考虑进行容器化。

    ![实践模拟](https://www.dailydoseofds.com/content/images/2025/08/image-32.png)

    如果你还没有学习第2部分，我们强烈建议先复习它，因为它建立了理解我们即将涵盖的材料所必需的概念基础。

    你可以在下面找到它：

    [The Full MLOps Blueprint: The Machine Learning System Lifecycle](https://www.dailydoseofds.com/mlops-crash-course-part-2/)

    在本章中，我们将探索ML系统的可重现性和版本控制，重点关注关键理论细节和实践实现（在需要时）。

    一如既往，每个概念都将通过清晰的示例和演练来解释，以培养清晰的理解。

    让我们开始吧！

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 引言

    我们在整个过程中都涉及到的一个主题，但值得重点关注的是可重现性。

    顾名思义，可重现性意味着你可以重复一个实验或过程并获得相同的结果。在ML中，这对信任和协作至关重要。

    ![可重现性重要性](https://www.dailydoseofds.com/content/images/2025/08/image-48.png)

    如果其他人（或未来的你）无法重现你的模型训练，就很难调试问题或改进它。

    ![调试困难](https://www.dailydoseofds.com/content/images/2025/08/image-45.png)

    可重现性也与版本控制密切相关，因为要重现实验，你需要确切知道使用了哪些代码、数据和参数。

    让我们分解为什么这很重要以及如何在生产ML系统中实现它。

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 为什么可重现性很重要

    通过我们到目前为止学到的内容，我们很好地理解了可重现性对生产级系统相当关键的事实。

    所以让我们扩展我们的理解，看看使可重现性和版本控制如此重要的关键因素。

    ### 调试和错误跟踪

    如果模型的性能突然下降，或者离线和在线行为之间存在差异，能够完全按照原样重现训练过程可以帮助查明原因。

    例如，是代码更改吗？库的新版本？不同的随机种子？没有可重现性，你实际上是在追逐一个移动的目标。

    ![调试挑战](https://www.dailydoseofds.com/content/images/2025/08/image-47.png)

    ### 协作

    在团队中，一个工程师可能想要重新运行另一个人的实验来验证结果或在此基础上构建。如果它不可重现，就会减慢进度。

    ![协作问题](https://www.dailydoseofds.com/content/images/2025/08/image-46.png)

    由于缺乏可重现性导致的协作问题

    重现某人的工作应该像拉取他们的代码和数据并运行脚本一样简单，而不是"你使用了什么环境？"的猜谜游戏。

    ### 法规和合规性

    在某些行业，如医疗保健、金融或自动驾驶汽车，你可能需要证明模型是如何构建的，以及它的行为是一致的。

    例如，银行可能需要向监管机构展示导致信用风险模型的确切训练程序，并且在相同数据上再次运行它会产生相同的结果。

    👉

    如果模型决策受到质疑（比如，有人指控它有偏见），你需要重新创建该决策是如何产生的。

    ### 连续性

    人员变动会发生，也许模型的原始作者离开了公司。

    如果过程有良好的文档记录且可重现，下一个人可以接手。如果没有，组织就有失去锁定在该模型中的"知识"的风险。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 生产问题

    我们经常定期重新训练模型。如果新模型版本比上一个版本表现更差，可重现性会有所帮助。你可以比较运行，因为你已经对所有成分进行了版本控制。

    ![模型版本比较](https://www.dailydoseofds.com/content/images/2025/08/image-50-1.png)

    此外，如果你需要回滚到以前的模型，你应该理想地重新训练它（如果数据已更改）或至少拥有确切的工件。版本控制允许你这样做。

    ![模型回滚](https://www.dailydoseofds.com/content/images/2025/08/image-49.png)

    现在我们了解了可重现性的重要性，让我们首先探索与之相关的挑战，然后深入了解如何将其纳入我们的系统。

    ---

    ## 可重现性的挑战

    与纯软件不同，ML的结果可能依赖于随机性（神经网络中的初始权重、随机训练-测试分割等）。如果不加以控制，使用相同代码/数据的两次运行仍可能产生略有不同的模型。

    正如我们在第1部分中看到的，ML是"部分代码，部分数据"。你需要对两者进行版本控制。

    ![代码和数据版本控制](https://www.dailydoseofds.com/content/images/2025/08/image-51.png)

    数据很大，所以你不能轻易地将数据集放入Git（另外，数据可能随时间更新）。

    ![数据版本控制挑战](https://www.dailydoseofds.com/content/images/2025/08/image-52.png)

    环境很重要——库版本、硬件（很少，不同GPU上的浮点精度等差异可能导致差异）、操作系统等。如果你的训练代码依赖于某些系统特定行为，那就是可重现性风险。

    ML模型和管道有很多移动部件，如模型超参数、特征管道、预处理步骤等。

    很容易有没有完全跟踪的"实验"。例如，你在笔记本中手动调整了某些内容并忘记了，导致一个难以复制的一次性模型。

    ![实验跟踪挑战](https://www.dailydoseofds.com/content/images/2025/08/image-53-1.png)

    接下来，让我们看看一些最佳实践，这些实践可以帮助我们解决这些挑战，并了解如何有效地将可重现性和版本控制纳入ML系统。

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 可重现性和版本控制的最佳实践

    现在我们对可重现性的重要性和它带来的挑战有了扎实的理解，让我们探索帮助我们充分利用可重现性和版本控制好处的最佳实践。

    ### 确保确定性过程

    如果需要精确的可重现性，在训练代码中设置随机种子。大多数ML库都允许这样做（例如，`np.random.seed(0)`、`random.seed(0)`，对于TensorFlow/PyTorch等框架，有方法为它们的随机操作固定种子）。

    但是，请注意，由于竞争条件或精度降低，某些并行或GPU操作本质上是非确定性的。如果你需要位对位相同的结果，你可能必须牺牲一些性能（许多框架都有"确定性"模式，但稍微慢一些）。

    在实践中，容差内的可重现性通常就足够了，即模型可能不是位相同的，但应该有类似的性能和类似的输出。尽管如此，控制随机性来源（数据洗牌顺序、权重初始化等）是良好的实践。

    此外，如果使用多线程，请注意执行顺序可能会有所不同。通常，固定种子并为关键部分使用单线程可以改善确定性。

    经验法则：如果有人在同一台机器上运行你的训练管道两次，它应该产生有效相同的模型（或指标）。如果没有，记录这一点（也许是由于非确定性），至少确保指标相同。

    ### 代码的版本控制

    这是不可协商的。从数据准备脚本到模型训练代码的所有代码都应该在Git（或另一个版本控制系统，缩写为VCS）中。每个实验都应该理想地与Git提交或标签相关联。

    ![代码版本控制](https://www.dailydoseofds.com/content/images/2025/08/image-54.png)

    这样，你就确切知道哪些代码产生了哪个模型。在实践中，团队经常在模型的元数据中包含Git提交哈希。这允许从模型追溯到代码。

    代码版本控制在软件工程中是众所周知的；ML只需要将这种严格性扩展到其他工件。

    ### 版本数据

    这个更棘手但至关重要。至少，如果你正在重新训练模型，保存使用的确切数据的快照或引用。如果训练数据存在于数据库中并且不断变化，你可能需要对其进行快照。

    如果不管理，你可能最终会对同一数据的不同变体感到困惑，不了解确切使用了哪一个。

    ![数据版本控制](https://www.dailydoseofds.com/content/images/2025/08/image-64.png)

    理想情况下，我们使用像DVC（数据版本控制）这样的工具，它将Git工作流扩展到数据和模型。DVC不在Git中存储实际数据，但存储哈希/引用，以便数据文件可以在外部进行版本控制（例如，在云存储上），同时仍然与Git提交相关联。

    ![DVC工作流](https://www.dailydoseofds.com/content/images/2025/08/image-55-1-1-1.png)

    例如，你可以使用DVC跟踪你的`train.csv`，当你提交时，DVC记录该文件的哈希（或指向云对象的指针）。稍后，即使文件很大，你也可以重现那个确切的文件。

    DVC提供"类似Git的体验来组织你的数据、模型和实验"。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 测试可重现性

    这可能听起来很抽象，但要有一个验证可重现性的过程。例如，在训练和保存模型后，你可以做一个快速测试：加载模型并在已知测试输入上运行它，看看结果是否符合预期。

    这确保模型文件没有损坏，环境确实可以产生相同的输出。另一个想法是，如果你用相同的数据重新训练模型（也许用不同的随机种子或在一些良性更改后），确保指标在同一范围内。

    如果它们差异很大，就有问题。要么是错误，要么是不稳定的训练过程。对于关键模型，你甚至可能有一个"可重现性测试"，你尝试重新运行旧的训练作业（在存档数据上）来看看是否得到相同的结果，作为CI的一部分。

    这并不常见（因为可能很昂贵），但在概念上类似于软件中的回归测试。

    ### 跟踪实验和元数据

    使用实验跟踪器（如MLflow）记录实验及其运行的所有内容。

    ![MLflow实验跟踪](https://www.dailydoseofds.com/content/images/2025/08/image-63.png)

    ![MLflow界面](https://www.dailydoseofds.com/content/images/2025/08/image-59.png)

    这通常意味着当你执行训练脚本时，它会记录：

    - 唯一的运行ID
    - 使用的参数（超参数、训练轮次等）
    - 指标（准确性、各轮次的损失等）
    - 代码版本（通过Git哈希，如上所述）
    - 数据版本（也许是DVC数据哈希或数据集ID）
    - 模型工件或对它的引用
    - 可能的环境信息（库版本）

    有了这些信息，你就有了每个实验的记录。如果运行#42是最好的并成为生产，任何人都可以检查运行#42的详细信息，理想情况下通过检出相同的代码并用相同的数据和参数重新运行来重现它。

    👉

    一些工具甚至允许克隆实验运行（如W&B有一个功能可以轻松同步代码并运行它）。更简单的是，只是有一个结构化的日志意味着手动重现也是可行的。

    请注意，MLflow中的实验是运行的命名集合，其中每个运行代表机器学习工作流或训练过程的特定执行。

    ![MLflow实验概念](https://www.dailydoseofds.com/content/images/2025/08/image-60.png)

    进一步详细说明，运行是特定实验内机器学习工作流的单次执行。

    它封装了该特定执行的所有详细信息，包括运行期间产生的代码、参数、指标和工件。

    ![MLflow运行概念](https://www.dailydoseofds.com/content/images/2025/08/image-61.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 版本模型工件

    每次训练可能投入生产的模型时，给它一个版本号或ID。在模型注册表中注册它。

    ![模型注册表](https://www.dailydoseofds.com/content/images/2025/08/image-58.png)

    像MLflow这样的工具提供了一个中央位置来管理具有版本和阶段的模型（例如，"v1–staging"、"v1–production"）。

    ![MLflow模型注册表](https://www.dailydoseofds.com/content/images/2025/08/image-56.png)

    模型注册表条目包含模型工件和元数据，如谁创建了它、何时创建，通常还有对实验或代码的引用。例如，MLflow的模型注册表提供UI和API来在阶段之间转换模型并保持历史记录。

    ![模型注册表UI](https://www.dailydoseofds.com/content/images/2025/08/image-57.png)

    它还存储血缘，即哪个运行（使用哪些参数和数据）产生了该模型。通过使用这样的注册表，你确保即使今天部署模型v5，如果需要，你仍然可以获取模型v3，并且你确切知道每个版本是什么。

    ### 数据和模型血缘记录

    当训练模型时，记录对确切数据的引用。例如，如果你使用带有分区的数据湖，注意哪个分区或时间戳。如果你查询数据库，在日志中包含查询或数据校验和。

    一些高级设置使用数据血缘工具（如通过管道跟踪数据来源）。对大多数人来说，即使只是记录"使用了大小为Y字节、校验和为Z的文件X"也很棒。它允许你稍后验证你有相同的文件。如果使用DVC，DVC提交ID充当链接。

    ### 环境管理

    使用工具捕获软件环境：

    - 使用`requirements.txt`或`environment.yml`（对于Conda）来固定训练和推理所需的库版本。
    - 避免"浮动"依赖项（如只说`pandas`而不指定版本），因为库的更新可能会改变行为。
    - 如果可能，容器化：Docker镜像可以作为环境的精确快照。你甚至可以对Docker镜像进行版本控制（如`my-train-env:v1`）。
    - 如果不使用容器，使用虚拟环境来隔离依赖项。这样，一年后使用相同的`requirements.txt`运行管道可以（希望）重新创建所需的环境。
    - 基础设施即代码：如果你的训练涉及启动某些云实例或使用特定硬件，将其脚本化。这样，即使是意外使用具有不同功能的GPU等基础设施差异也不太可能潜入。

    ### 权衡

    值得注意的是，绝对可重现性（位对位）有时是不必要的严格。在许多情况下，我们关心的是性能或行为在容差范围内是可重现的，而不是确切的权重。

    例如，如果训练深度网络一次给出0.859的准确性，下次给出0.851，在有用性方面这实际上是相同的。试图获得相同的权重可能是过度的。

    但是，如果你一次得到0.88，另一次在据说相同的设置下得到0.80，这表明有问题（如也许有些你没有控制的东西改变了）。

    所以要追求一致性，但不要因为非确定性引起的微小差异而恐慌。从商业角度来看，重要的是质量的一致性。也就是说，如果你可以轻松实现位对位可重现性，就这样做，因为它简化了调试，但要意识到如前所述的非确定性来源。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    总结一切，在实践中，实现可重现性需要纪律和文化，就像需要工具一样。你必须灌输这样的习惯："我提交了那个代码吗？我标记了数据版本吗？我在记录运行吗？"

    有时，快速实验诱使人们快速做事而不跟踪，但这可能导致以后花费更多时间来理清做了什么。一个常见的格言是："如果它没有被记录或版本控制，它就没有发生。"

    所以团队经常采用检查清单或自动化来强制执行这一点（例如，训练脚本自动记录到MLflow，所以你不能忘记）。

    总的来说，可重现性和版本控制是使ML项目在长期内可管理和可靠的因素。它们将ML从一次性艺术转变为工程学科。

    通过对一切进行版本控制（代码、数据、模型、配置）并使用工具来帮助跟踪它，我们为我们的模型创建了审计跟踪。这不仅建立了信心（我们可以调试，我们可以改进，我们可以信任生产中运行的内容，因为我们知道它来自哪里），还节省了时间。

    ![可重现性价值](https://www.dailydoseofds.com/content/images/2025/08/image-62.png)

    正如俗话说，"如果它不可重现，它就不是科学。"在MLOps中，如果它不可重现，它在生产中就不会稳健。

    现在我们已经涵盖了关键原则，让我们通过一些实践模拟将这些想法付诸实践。

    ---

    ## PyTorch模型训练循环和模型持久化

    在这个例子中，我们说明了神经网络的简单PyTorch训练循环，包括如何处理可重现性（使用种子）以及如何保存和加载模型权重。

    我们在这里不会使用实验跟踪器以保持简洁，但会显示记录打印和保存。在实践中，你会将其与W&B或MLflow等工具集成。

    👉

    不用担心，我们稍后会在单独的示例中介绍如何使用MLflow进行实验跟踪。

    ### 项目设置

    这个项目的代码旨在使用Google Colab运行。我们建议在Colab中上传`.ipynb`笔记本，并从那里运行它。

    在这里下载笔记本：

    [MLOps-Reproducibility-PyTorch.ipynb](https://www.dailydoseofds.com/content/files/2025/08/MLOps-Reproducibility-PyTorch.zip)

    ### 代码实现

    让我们看看一个完整的PyTorch训练示例，展示可重现性的最佳实践：

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import random
    import os
    from datetime import datetime
    import json

    # 设置随机种子以确保可重现性
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置种子
    set_seed(42)

    # 创建简单的神经网络
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # 生成合成数据
    def generate_data(n_samples=1000, n_features=10, seed=42):
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        y = (X.sum(axis=1) > 0).astype(int)  # 简单的二分类
        return X.astype(np.float32), y

    # 训练函数
    def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
        model.train()
        train_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        return train_losses
    ```

    这个示例展示了如何在PyTorch中实现可重现的训练过程，包括种子设置、模型定义和训练循环。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 模型保存和加载

    让我们继续展示如何正确保存和加载模型，包括元数据：

    ```python
    # 模型保存函数
    def save_model_with_metadata(model, model_path, metadata):
        # 创建保存目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 保存模型状态字典
        torch.save(model.state_dict(), model_path)

        # 保存元数据
        metadata_path = model_path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"模型已保存到: {model_path}")
        print(f"元数据已保存到: {metadata_path}")

    # 模型加载函数
    def load_model_with_metadata(model_class, model_path, model_params):
        # 加载元数据
        metadata_path = model_path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 创建模型实例
        model = model_class(**model_params)

        # 加载模型权重
        model.load_state_dict(torch.load(model_path))

        print(f"模型已从 {model_path} 加载")
        print(f"训练时间: {metadata['training_time']}")
        print(f"最终损失: {metadata['final_loss']:.4f}")

        return model, metadata

    # 主训练脚本
    def main():
        # 设置参数
        config = {
            'input_size': 10,
            'hidden_size': 64,
            'output_size': 1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'seed': 42
        }

        # 设置种子
        set_seed(config['seed'])

        # 生成数据
        X, y = generate_data(n_samples=1000, n_features=config['input_size'])

        # 创建数据加载器
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # 创建模型
        model = SimpleNN(config['input_size'], config['hidden_size'], config['output_size'])

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # 训练模型
        print("开始训练...")
        start_time = datetime.now()
        train_losses = train_model(model, train_loader, criterion, optimizer, config['num_epochs'])
        end_time = datetime.now()

        # 准备元数据
        metadata = {
            'config': config,
            'training_time': str(end_time - start_time),
            'final_loss': train_losses[-1],
            'model_architecture': str(model),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__
        }

        # 保存模型
        model_path = 'models/simple_nn_model.pth'
        save_model_with_metadata(model, model_path, metadata)

        return model, metadata

    # 运行训练
    if __name__ == "__main__":
        trained_model, training_metadata = main()
    ```

    这个完整的示例展示了：
    - 如何设置随机种子确保可重现性
    - 如何保存模型权重和训练元数据
    - 如何加载模型并验证其配置
    - 如何记录训练过程的关键信息
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## MLflow实验跟踪和模型注册

    现在让我们看看如何使用MLflow来跟踪实验和注册模型。MLflow是一个开源平台，用于管理ML生命周期，包括实验、可重现性和部署。

    ### MLflow基础概念

    在深入代码之前，让我们了解MLflow的核心概念：

    - **实验（Experiment）**：相关运行的集合，通常对应一个ML项目或问题
    - **运行（Run）**：单次模型训练执行，包含参数、指标和工件
    - **工件（Artifacts）**：与运行相关的文件，如模型、图表、数据等
    - **模型注册表（Model Registry）**：用于管理模型版本和生命周期的中央存储库

    ### 安装和设置

    ```bash
    pip install mlflow
    pip install scikit-learn pandas numpy matplotlib
    ```

    ### 使用MLflow跟踪实验

    让我们创建一个使用MLflow跟踪的完整示例：

    ```python
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    # 设置MLflow跟踪URI（本地文件系统）
    mlflow.set_tracking_uri("file:./mlruns")

    # 创建或设置实验
    experiment_name = "可重现性演示实验"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    def run_ml_experiment(n_estimators=100, max_depth=10, random_state=42):
        with mlflow.start_run() as run:
            # 记录参数
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("algorithm", "RandomForest")

            # 生成数据
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=10,
                n_redundant=10,
                random_state=random_state
            )

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            # 训练模型
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # 预测
            y_pred = model.predict(X_test)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # 记录指标
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("training_time_seconds", training_time)

            # 记录模型
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="RandomForestDemo"
            )

            # 创建并记录特征重要性图
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()

            # 记录数据集信息
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X.shape[1])

            print(f"运行ID: {run.info.run_id}")
            print(f"准确率: {accuracy:.4f}")
            print(f"F1分数: {f1:.4f}")

            return run.info.run_id, model

    # 运行多个实验
    print("运行实验1: 默认参数")
    run_id_1, model_1 = run_ml_experiment()

    print("\n运行实验2: 更多树")
    run_id_2, model_2 = run_ml_experiment(n_estimators=200)

    print("\n运行实验3: 更深的树")
    run_id_3, model_3 = run_ml_experiment(max_depth=20)
    ```

    这个示例展示了如何使用MLflow来：

    - 跟踪实验参数和指标
    - 记录模型工件
    - 保存可视化结果
    - 自动注册模型到模型注册表
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 模型注册表操作

    MLflow模型注册表提供了一个中央位置来管理模型的生命周期。让我们看看如何使用它：

    ```python
    from mlflow.tracking import MlflowClient

    # 创建MLflow客户端
    client = MlflowClient()

    # 列出所有注册的模型
    registered_models = client.list_registered_models()
    print("注册的模型:")
    for model in registered_models:
        print(f"- {model.name}")

    # 获取特定模型的版本
    model_name = "RandomForestDemo"
    model_versions = client.get_registered_model(model_name)
    print(f"\n{model_name} 的版本:")
    for version in model_versions.latest_versions:
        print(f"- 版本 {version.version}: {version.current_stage}")

    # 转换模型阶段
    def transition_model_stage(model_name, version, stage):
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"模型 {model_name} 版本 {version} 已转换到 {stage} 阶段")

    # 示例：将最新版本转换到Staging
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    transition_model_stage(model_name, latest_version.version, "Staging")

    # 加载特定阶段的模型
    def load_model_from_registry(model_name, stage="Production"):
        model_uri = f"models:/{model_name}/{stage}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"成功加载 {model_name} 的 {stage} 版本")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

    # 尝试加载Staging版本
    staging_model = load_model_from_registry(model_name, "Staging")

    # 模型版本比较
    def compare_model_versions(model_name):
        versions = client.get_latest_versions(model_name)

        print(f"\n{model_name} 版本比较:")
        print("-" * 50)

        for version in versions:
            run = client.get_run(version.run_id)
            metrics = run.data.metrics
            params = run.data.params

            print(f"版本 {version.version} ({version.current_stage}):")
            print(f"  准确率: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  F1分数: {metrics.get('f1_score', 'N/A'):.4f}")
            print(f"  参数: n_estimators={params.get('n_estimators', 'N/A')}, "
                  f"max_depth={params.get('max_depth', 'N/A')}")
            print()

    # 比较模型版本
    compare_model_versions(model_name)
    ```

    ### 启动MLflow UI

    要查看实验和模型注册表的Web界面，运行：

    ```bash
    mlflow ui --backend-store-uri file:./mlruns
    ```

    然后在浏览器中访问 `http://localhost:5000` 来查看：

    - 实验列表和运行详情
    - 参数、指标和工件的比较
    - 模型注册表和版本管理
    - 模型血缘和元数据

    ### DVC数据版本控制示例

    除了MLflow，我们还可以使用DVC来版本控制数据：

    ```bash
    # 初始化DVC
    dvc init

    # 添加数据文件到DVC跟踪
    dvc add data/train.csv

    # 提交到Git
    git add data/train.csv.dvc data/.gitignore
    git commit -m "添加训练数据"

    # 推送数据到远程存储
    dvc push

    # 在另一台机器上拉取数据
    dvc pull
    ```

    这样，我们就建立了一个完整的可重现性工作流：
    - 使用Git进行代码版本控制
    - 使用DVC进行数据版本控制
    - 使用MLflow进行实验跟踪和模型注册
    - 使用Docker进行环境版本控制
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    在MLOps和LLMOps速成课程的第三部分中，我们学习了可以说是机器学习系统最重要但经常被低估的支柱之一：可重现性和版本控制。

    我们首先基于为什么可重现性很重要来建立我们的理解，不仅作为理论理想，而且作为长期模型可靠性的实际要求。

    我们探索了可重现性在正确执行时如何在ML生命周期中建立信任，并使你今天所做的工作能够应对明天的未知。

    从那里，我们走过了使机器学习中的可重现性如此困难的细致挑战，特别是随机性、可变数据、环境漂移和临时实验，所有这些都将ML与传统软件工程区分开来。

    为了应对这些挑战，我们介绍了全方位的最佳实践：

    - 使用Git进行代码版本控制并将实验与特定提交相关联
    - 利用DVC等工具进行数据集和模型工件版本控制
    - 使用MLflow等实验跟踪器记录参数、指标和模型血缘
    - 通过种子固定、容器化和依赖项固定来控制随机性和环境

    但我们没有止步于理论。

    通过实践演练，我们演示了如何：

    - 在PyTorch中构建可重现的训练循环
    - 确定性地保存和加载模型检查点
    - 使用Git + DVC对代码和数据进行版本控制
    - 使用MLflow跟踪实验和注册模型，通过代码和UI

    这些示例中的每一个都强化了一个核心思想：可重现性不是你在顶部撒上的功能；它是融入开发过程的心态。

    我们也承认了权衡。我们看到绝对的位对位可重现性并不总是必要的。在生产中更重要的是一致的行为、稳健的性能，以及模型行为方式的可追溯性。

    本章中最强烈的信息之一是，可重现性不是增加开销；它是节省未来时间、避免回归错误、实现协作和保持问责制。它是实验和生产之间的桥梁，是可扩展、可审计和高影响ML系统的基础。

    当我们展望本系列的未来章节时，我们将在这个坚实的基础上构建，探索：

    - 扩展可重现性和版本控制的工具
    - 为ML系统量身定制的CI/CD工作流
    - 来自行业的真实案例研究
    - 生产中的监控和观察
    - LLMOps的特殊考虑
    - 结合生命周期所有元素的完整端到端示例

    正如在整个系列中持续强调的，本课程的目标是大量倾向于理论

    实现细节可能因用例、规模和行业而异。但如果你深入理解底层系统设计和生命周期原则，你将有能力导航任何MLOps堆栈或适应任何LLMOps场景。

    所以，随着我们前进，期待看到理论、方法和模拟的持续融合，弥合实验和生产之间的差距。目标是帮助你培养成熟的、以系统为中心的心态，将机器学习不视为独立工件，而是更广泛软件生态系统的活跃部分。

    我们还要花一分钟强调，在系列的后期，一旦我们过渡到LLMOps，我们还将探索当应用于基于LLM的系统时策略如何演变，其中提示、嵌入和上下文窗口成为新的管道，模型行为可能基于非代码参数而变化。

    ---

    ## 关键要点

    🎯 **可重现性的核心价值**：

    - 建立对ML系统的信任
    - 简化调试和问题追踪
    - 促进团队协作
    - 满足合规和审计要求

    🔧 **实现可重现性的工具链**：

    - **Git**: 代码版本控制
    - **DVC**: 数据和模型版本控制
    - **MLflow**: 实验跟踪和模型注册
    - **Docker**: 环境版本控制

    📋 **最佳实践检查清单**：

    - ✅ 设置随机种子
    - ✅ 版本控制所有代码
    - ✅ 跟踪数据版本
    - ✅ 记录实验参数和指标
    - ✅ 保存模型工件和元数据
    - ✅ 管理环境依赖
    - ✅ 测试可重现性

    💡 **记住**：可重现性不是开销，而是投资。它为未来的你和你的团队节省时间，并使ML从艺术转变为工程学科。
    """
    )
    return


if __name__ == "__main__":
    app.run()

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
    # 使用Modelbit直接从Jupyter Notebook部署、版本控制和管理ML模型

    ## 引言

    在最近的深入研究中，我们主要专注于培养能够帮助我们开发大型机器学习（ML）项目的技能。

    例如，在最近关于"**模型压缩**"的深入研究中，我们学习了许多技术来大幅减少模型的大小，使其更适合生产环境。

    ![模型压缩](https://www.dailydoseofds.com/content/images/2023/09/image-142.png)

    ![Logo](https://www.dailydoseofds.com/content/images/size/w256h256/format/png/2023/06/logo-subsatck2-1.svg)

    在上述文章中，我们看到这些技术如何让我们减少原始模型的延迟和大小，这直接有助于：

    - 降低计算成本
    - 减少模型占用空间
    - 由于低延迟而改善用户体验...

    **...所有这些都是企业的关键指标。**

    然而，仅仅学习模型压缩技术是不够的。

    在大多数情况下，我们只有在模型打算为最终用户服务时才会进行模型压缩。

    ![部署重要性](https://www.dailydoseofds.com/content/images/2023/09/image-204.png)

    而这只有在我们知道如何在生产中部署和管理机器学习时才可能实现。

    因此，在学习了模型压缩技术之后，我们准备学习下一个关键技能——**部署**。

    在我看来，许多人将部署仅仅视为"部署"——在某处托管模型，获得API端点，将其集成到应用程序中，就完成了！

    **但这几乎从来不是这样的。**

    这是因为，在现实中，部署后必须做很多事情来确保模型的可靠性和性能。

    它们是什么？让我们来了解一下！

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 部署后的考虑因素

    部署是ML项目生命周期中的关键阶段。这是真实用户将依赖你的模型预测的阶段。

    然而，重要的是要认识到部署不是最终目的地。

    部署模型后，必须解决几个关键考虑因素以确保其可靠性和性能。

    让我们来了解它们。

    ### #1) 版本控制

    版本控制对所有开发过程都至关重要。它允许开发人员跟踪软件随时间的变化（代码、配置、数据等）。

    在数据团队的背景下，版本控制在部署模型时可能特别关键。

    例如，通过版本控制，人们可以精确识别什么发生了变化、何时发生变化以及谁进行了变化——这在试图诊断和修复部署过程中出现的问题或模型在部署后开始表现不佳时是关键信息。

    ![版本控制重要性](https://www.dailydoseofds.com/content/images/2023/09/image-164.png)

    这回到了我们在最近的深入研究中讨论的内容——"_机器学习应该具有任何软件工程领域的严格性。_"

    如果模型开始表现不佳，基于git的功能允许我们快速回滚到模型的先前版本。

    还有许多其他好处。

    #### #1.1) 协作

    随着数据科学项目变得越来越大，有效的协作变得越来越重要。

    团队中的某人可能正在为模型识别更好的特征，而其他人可能负责微调超参数或优化部署基础设施。

    ![团队协作](https://www.dailydoseofds.com/content/images/2023/09/image-168.png)

    众所周知，通过版本控制，团队可以在相同的代码库/数据上工作，改进相同的模型，而不会相互干扰。

    此外，人们可以轻松跟踪变化、审查彼此的工作并解决冲突（如果有的话）。

    #### #1.2) 可重现性

    可重现性是构建可靠机器学习的关键方面之一。

    想象一下：在一个系统上工作但在另一个系统上不工作的东西反映了糟糕的可重现性实践。

    你可能想知道为什么它很重要？

    它确保结果可以被其他人复制和验证，这提高了我们工作的整体可信度。

    ![可重现性](https://www.dailydoseofds.com/content/images/2023/09/image-170.png)

    版本控制允许我们跟踪用于产生特定结果的确切代码版本和配置，使将来更容易重现结果。

    这对于许多人可能使用的开源数据项目特别有用。

    #### #1.3) 持续集成和持续部署（CI/CD）

    CI/CD使团队能够快速高效地构建、测试和部署代码。

    在机器学习中，持续集成（CI）可能涉及在ML模型的更改提交到代码存储库后立即自动构建和测试这些更改。

    ![CI过程](https://www.dailydoseofds.com/content/images/2023/09/image-205.png)

    在持续部署（CD）中，目标可以是在模型通过测试后反映新的更改。

    ![CD过程](https://www.dailydoseofds.com/content/images/2023/09/image-206.png)

    因此，它应该无缝地将更改更新到生产环境，使模型的最新版本对最终用户可用。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### #2) 模型日志记录

    模型日志记录是部署后ML操作的另一个关键方面。

    顾名思义，日志记录涉及捕获和存储有关模型性能、资源利用、预测、输入数据、延迟等的相关信息。

    ![模型日志记录](https://www.dailydoseofds.com/content/images/2023/09/image-172.png)

    模型日志记录很重要的原因有很多，这是绝不应该被忽视的事情。

    为了更好地理解，想象你已经部署了一个模型，它正在为最终用户服务。

    一旦部署，在生产中几乎不可能什么都不出错，**特别是在数据方面**！

    让我们详细了解一下。

    #### #2.1) 概念漂移

    **概念漂移**发生在目标变量或作为模型输入发送的输入特征的统计属性随时间变化时。

    简单来说，模型输入和输出之间的关系会演变，如果不加以解决，会使模型随时间变得不那么准确。

    ![概念漂移](https://www.dailydoseofds.com/content/images/2023/09/image-207.png)

    概念漂移可能由于各种原因而发生，例如：

    - 用户行为的变化
    - 数据源的转移
    - 底层数据生成过程的改变

    例如，想象你正在构建一个垃圾邮件分类器。你在几个月内收集的数据集上训练模型。

    ![垃圾邮件分类器](https://www.dailydoseofds.com/content/images/2023/09/image-210.png)

    最初，模型表现良好，准确地分类垃圾邮件和非垃圾邮件。

    然而，随着时间的推移，电子邮件垃圾邮件技术会演变。

    出现了具有不同关键词、结构和技术的新类型垃圾邮件。

    "垃圾邮件"底层概念的这种变化代表了概念漂移。

    这就是为什么重要的是要有定期重新训练或持续训练策略。

    如果你的模型没有定期用最新数据重新训练，它可能开始错误分类新类型的垃圾邮件，导致性能下降。

    #### #2.2) 协变量偏移

    💡

    术语'协变量'指的是模型的特征。

    **协变量偏移**是概念漂移的一种特定类型，当数据中输入特征（协变量）的分布随时间变化时发生，但目标变量和输入之间的真实关系保持不变。

    换句话说，输入特征和目标变量之间的真实（或自然）关系保持恒定，但输入特征的分布发生偏移。

    例如，考虑这是真实关系，它是非线性的：

    ![真实关系](https://www.dailydoseofds.com/content/images/2023/09/image-213.png)

    基于观察到的训练数据，我们最终学习了一个线性关系：

    ![学习关系](https://www.dailydoseofds.com/content/images/2023/09/image-216.png)

    然而，在部署后推理时，输入样本的分布与观察到的分布不同：

    ![分布偏移](https://www.dailydoseofds.com/content/images/2023/09/image-217.png)

    这导致模型性能差，因为模型是在数据的一个分布上训练的，但现在它正在不同的分布上进行测试或部署。

    解决协变量偏移的方法包括重新加权训练数据或使用域适应技术来对齐源分布和目标分布。

    💡

    在某种程度上，批量归一化是神经网络中协变量偏移的补救措施。

    例如，假设你正在构建一个天气预报模型。你使用来自特定地区的历史天气数据训练模型，训练数据包括温度、湿度和风速等特征。

    然而，当你将模型部署到具有不同气候的不同地区时，这些特征的分布可能会显著偏移。

    例如，新地区的温度范围和湿度水平可能与训练数据中的那些大不相同。这种协变量偏移可能导致你的模型在新环境中做出不准确的预测。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### #2.3) 非平稳性

    在构建统计模型时，我们通常假设样本是同分布的。

    **非平稳性**指的是样本的**概率分布**以非系统性或不可预测的方式随时间演变的情况。

    ![非平稳性](https://www.dailydoseofds.com/content/images/2023/09/image-218.png)

    这可以包括各个方面，包括数据分布、趋势、季节性或其他模式的变化。

    **非平稳性对机器学习模型来说可能是具有挑战性的，因为它们通常是在假设数据分布保持恒定的情况下训练的。**

    当数据变得非平稳时，模型的性能可能会下降，因为它们是基于不再适用的假设进行训练的。

    ### #3) 模型监控

    模型监控是部署后ML操作的另一个关键方面。

    一旦模型部署到生产环境中，持续监控其性能、行为和健康状况就变得至关重要。

    监控帮助我们：

    - **检测性能下降**：识别模型何时开始表现不佳
    - **发现异常**：检测异常的预测模式或输入数据
    - **跟踪资源使用**：监控CPU、内存和延迟指标
    - **验证数据质量**：确保输入数据符合预期格式和范围

    ![模型监控](https://www.dailydoseofds.com/content/images/2023/09/image-172.png)

    有效的监控系统应该包括：

    1. **实时警报**：当关键指标超出阈值时立即通知
    2. **仪表板**：可视化模型性能和系统健康状况
    3. **日志分析**：详细记录预测、输入和系统事件
    4. **性能基准**：与历史性能进行比较

    ---

    ## Modelbit简介

    现在我们了解了部署后的关键考虑因素，让我们探索一个能够简化这些复杂性的工具：**Modelbit**。

    Modelbit是一个平台，允许数据科学家直接从Jupyter Notebook部署、版本控制和管理机器学习模型。

    ### 为什么选择Modelbit？

    传统的ML部署通常涉及：

    1. **复杂的基础设施设置**：配置服务器、容器、负载均衡器
    2. **DevOps专业知识**：需要了解Docker、Kubernetes、云服务
    3. **长开发周期**：从模型到生产API的漫长过程
    4. **维护负担**：持续的服务器管理和更新

    Modelbit通过以下方式简化了这个过程：

    ✅ **一行代码部署**：直接从Notebook部署模型

    ✅ **自动版本控制**：跟踪模型版本和变更

    ✅ **内置监控**：实时性能和使用情况跟踪

    ✅ **零基础设施管理**：无需设置服务器或容器

    ✅ **Git集成**：与现有开发工作流程无缝集成

    ### Modelbit工作流程

    使用Modelbit的典型工作流程如下：

    ```
    1. 在Jupyter Notebook中训练模型
    2. 定义推理函数
    3. 指定环境依赖
    4. 一行代码部署
    5. 获得生产就绪的API端点
    6. 监控和管理部署
    ```

    让我们通过一个实际例子来看看这个过程！
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 使用Modelbit部署ML模型：分步指南

    让我们通过一个完整的例子来演示如何使用Modelbit部署机器学习模型。

    ### 步骤1：连接到Modelbit

    首先，我们需要将Jupyter内核连接到Modelbit。

    ```python
    # 安装Modelbit
    !pip install modelbit

    # 导入并连接
    import modelbit as mb

    # 连接到Modelbit（首次使用需要认证）
    mb.login()
    ```

    成功连接后，你会看到确认消息，表明Jupyter内核已成功连接到Modelbit。

    ### 步骤2：训练ML模型

    为了简单起见，让我们假设我们的模型是使用sklearn训练的线性回归模型。

    💡

    模型也可以是任何其他传统机器学习或深度学习模型。

    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    import numpy as np

    # 生成示例数据
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 验证模型
    print(f"模型训练完成，R²得分: {model.score(X, y):.4f}")
    ```

    ### 步骤3：定义推理函数

    参考我们的工作流程动画，下一步是定义推理函数。

    ![工作流程](https://www.dailydoseofds.com/content/images/2023/09/Screen-Recording-2023-09-22-at-2.37.09-PM.gif)

    顾名思义，这个函数必须包含在推理时执行的代码。因此，它将负责返回预测。

    ![推理函数](https://www.dailydoseofds.com/content/images/2023/09/image-174.png)

    我们必须在此方法中指定模型所需的输入参数。此外，我们可以给它任何我们想要的名称。

    对于我们的线性回归情况，推理函数可以如下：

    ```python
    def my_lr_deployment(input_x):
        \"\"\"
        推理函数: 接收输入并返回模型预测

        参数:
        input_x: 输入特征值(数字或数字列表)

        返回:
        预测值
        \"\"\"
        # 验证输入数据类型
        if not isinstance(input_x, (int, float, list, np.ndarray)):
            raise ValueError("输入必须是数字或数字列表")

        # 确保输入是正确的形状
        if isinstance(input_x, (int, float)):
            input_x = [[input_x]]
        elif isinstance(input_x, list):
            input_x = [input_x] if not isinstance(input_x[0], list) else input_x

        # 进行预测
        prediction = model.predict(input_x)

        # 返回预测结果
        return prediction.tolist()
    ```

    如上所示：

    - 我们定义了一个函数`my_lr_deployment()`
    - 接下来，我们将模型的输入指定为函数的参数（`input_x`）
    - 然后，我们验证输入的数据类型
    - 最后，我们返回模型的预测

    Modelbit的一个好处是，函数的每个依赖项（在这种情况下是`model`对象）都会自动pickle并与函数一起发送到生产环境。

    因此，我们可以在此方法中引用任何对象。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 步骤4：指定环境

    这是一个可选但推荐的步骤。

    明确指定我们在模型开发中使用的Python和开源库的确切版本可能是至关重要的。

    本质上，在推理期间（即，一旦模型为最终用户服务），模型将使用这些指定的版本进行预测。

    因此，确保我们在预测时使用与训练时相同的环境是很好的，以避免任何不必要的错误。

    在使用Modelbit部署的情况下，环境直接与部署API调用一起发送到Modelbit。

    让我们在下一步中了解这一点！

    ### 步骤5：部署模型

    最后，我们进行部署。

    使用Modelbit，部署就像下面这行Python代码一样简单：

    ```python
    # 基本部署
    mb.deploy(my_lr_deployment)
    ```

    完成！

    我们已经成功部署了模型，而且是直接从Jupyter Notebook！

    ![部署成功](https://miro.medium.com/v2/resize:fit:700/1*rf-qTr_2pEVHcal7hCLddQ.gif)

    在上述部署中，我们从未指定任何库版本。因此，在部署模型时，如果Modelbit使用的库版本与我们本地的不匹配，Modelbit会提示我们，如下所示：

    ![版本警告](https://www.dailydoseofds.com/content/images/2023/09/Screenshot-2023-09-22-at-4.05.01-PM.png)

    为了解决这个问题，我们可以在`mb.deploy()`本身中指定Python和库版本，如下所示：

    ```python
    # 指定环境的部署
    mb.deploy(
        my_lr_deployment,
        python_version="3.9",
        python_packages=[
            "scikit-learn==1.3.0",
            "numpy==1.24.3",
            "pandas==2.0.3"
        ]
    )
    ```

    ### 推理

    一旦我们的模型成功部署，它将出现在我们的Modelbit仪表板中。

    ![Modelbit仪表板](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d515957-c683-4aee-9e09-aadd82616686_1244x896.gif)

    如上所示，Modelbit提供了一个API端点。我们可以将其用于推理目的，如下所示：

    ```bash
    # 使用curl进行API调用
    curl -X POST https://your-workspace.modelbit.com/v1/my_lr_deployment/latest \\
         -H "Authorization: Bearer YOUR_API_KEY" \\
         -H "Content-Type: application/json" \\
         -d '{"data": [[1, 3.5], [2, 7.2], [3, 1.8]]}'
    ```

    在上述请求中，`data`是一个列表的列表。

    列表中的第一个数字（`1`）是输入ID。`ID`可以是你喜欢使用的任何标识符。ID后面的数字是函数参数。

    例如，对于我们的`my_lr_deployment(input_x)`方法，数据列表的列表将如下所示：

    ```python
    # 格式: [id, input_x]
    [
        [1, 3],
        [2, 5],
        [3, 9]
    ]
    ```

    让我们用上述输入调用API：

    ```python
    import requests
    import json

    # API端点
    url = "https://your-workspace.modelbit.com/v1/my_lr_deployment/latest"

    # 请求头
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }

    # 请求数据
    data = {
        "data": [
            [1, 3.5],
            [2, 7.2],
            [3, 1.8]
        ]
    }

    # 发送请求
    response = requests.post(url, headers=headers, json=data)

    # 解析响应
    if response.status_code == 200:
        predictions = response.json()
        print("预测结果:", predictions)
    else:
        print("错误:", response.status_code, response.text)
    ```

    调用部署的模型不仅限于`curl`。我们也可以在Python中使用`requests`库。

    完成！

    这就是我们如何使用Modelbit部署ML模型。

    总结一下，步骤如下图所示：

    ![部署步骤](https://www.dailydoseofds.com/content/images/2023/09/Screen-Recording-2023-09-22-at-2.37.09-PM.gif)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 为部署添加额外文件

    在使用模型进行推理之前，我们可能需要执行一些辅助代码，这些代码可能存储在不同的脚本中。

    例如，我们可能需要在将输入数据发送到模型之前对其进行预处理。

    让我们看看如何在部署中包含这些辅助文件。

    想象一下，函数的输入首先通过预处理函数，然后再输入到模型中。

    ```python
    # preprocessing.py 文件内容
    import numpy as np

    def process_input(raw_input):
        \"\"\"
        预处理输入数据
        \"\"\"
        # 标准化输入
        processed = (raw_input - np.mean(raw_input)) / np.std(raw_input)
        return processed

    def validate_input(input_data):
        \"\"\"
        验证输入数据
        \"\"\"
        if not isinstance(input_data, (list, np.ndarray)):
            raise ValueError("输入必须是列表或numpy数组")
        return True
    ```

    然后在推理函数中使用：

    ```python
    # 导入预处理模块
    import preprocessing

    def my_enhanced_deployment(input_x):
        \"\"\"
        增强的推理函数，包含预处理
        \"\"\"
        # 验证输入
        preprocessing.validate_input(input_x)

        # 预处理输入
        processed_input = preprocessing.process_input(input_x)

        # 确保正确的形状
        if isinstance(processed_input, (int, float)):
            processed_input = [[processed_input]]
        elif isinstance(processed_input, list):
            processed_input = [processed_input] if not isinstance(processed_input[0], list) else processed_input

        # 进行预测
        prediction = model.predict(processed_input)

        return prediction.tolist()
    ```

    如上所示：

    - 首先，我们导入`preprocessing` Python文件
    - 接下来，我们使用其中定义的`process_input()`方法
    - 最后，在处理输入后，我们返回模型的预测

    由于推理函数依赖于来自`preprocessing`文件的`process_input()`函数，我们必须确保在部署和后续推理期间此文件的可用性。

    我们可以在使用`mb.deploy()`进行模型部署调用时通过指定`extra_files`参数来做到这一点，如下所示：

    ```python
    # 部署时包含额外文件
    mb.deploy(
        my_enhanced_deployment,
        extra_files=["preprocessing.py"]
    )
    ```

    如果推理函数依赖于多个这样的文件，我们可以在`extra_files`参数中将多个文件指定为列表：

    ```python
    # 包含多个额外文件
    mb.deploy(
        my_enhanced_deployment,
        extra_files=["preprocessing.py", "utils.py", "validators.py"]
    )
    ```

    ### 多个部署的通用文件

    在上述演示中，我们添加了特定于单个部署的Python文件。

    然而，如果我们打算部署多个模型，并且它们都依赖于相同的实用程序/处理文件，我们将不得不在使用`mb.deploy()`进行的各个部署调用中重复添加这些文件。

    这如下所示：

    ```python
    # 重复的文件包含
    mb.deploy(model1_function, extra_files=["preprocessing.py", "utils.py"])
    mb.deploy(model2_function, extra_files=["preprocessing.py", "utils.py"])  # 重复！
    mb.deploy(model3_function, extra_files=["preprocessing.py", "utils.py"])  # 重复！
    ```

    我们可以通过添加所有部署都可以引用的通用文件来避免这种情况。这些共享文件可能包括：

    - 共享日志代码
    - 分析代码
    - 共享数据预处理助手（如我们之前所做的），等等

    我们可以使用`mb.add_common_files()`方法添加所有部署都可以使用的通用文件。

    ```python
    # 添加通用文件
    mb.add_common_files("preprocessing.py")  # 添加单个文件

    mb.add_common_files(["preprocessing.py", "utils.py"])  # 添加多个文件

    mb.add_common_files("utils/")  # 添加整个目录
    ```

    在上述演示中：

    - 第一个语句让我们将一个文件——`preprocessing.py`添加到`common`文件夹
    - 第二个语句让我们添加两个（或更多）通用文件
    - 最后一个语句允许我们添加`utils`目录中的所有文件

    与添加通用文件类似，你也可以使用`mb.delete_common_files()`方法删除通用文件。

    ```python
    # 删除通用文件
    mb.delete_common_files("preprocessing.py")
    mb.delete_common_files(["preprocessing.py", "utils.py"])
    ```

    💡

    通用文件在我们部署模型时被复制到部署中。更新通用文件不会更新现有部署，以防止通用文件的更改破坏过去的版本。要在部署中包含通用文件的最新版本，请重新部署它。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 版本控制和Git集成

    到目前为止，我们已经学会了：

    - 如何使用Modelbit直接从Jupyter Notebook部署机器学习模型
    - 我们必须遵循的端到端过程来使用Modelbit部署模型
    - 如何在部署时推送文件
    - 如何使通用文件在部署中可用
    - 如何删除通用文件

    但这就是全部吗？

    绝对不是！

    到目前为止，我们只学会了如何使用Modelbit API从Jupyter Notebook部署模型。

    但在后期阶段，我们可能需要更新模型（或代码/数据集）。

    在本文前面，我们讨论了为什么版本控制很重要，无论是从开发还是部署的角度。

    让我们看看如何在Modelbit的部署阶段利用基于git的功能。

    ### Git集成的好处

    Modelbit与Git的集成提供了几个关键优势：

    1. **代码版本控制**：跟踪模型代码的每个变更
    2. **协作开发**：团队成员可以协作开发和部署模型
    3. **回滚能力**：快速回滚到之前的模型版本
    4. **审计跟踪**：完整的变更历史记录
    5. **CI/CD集成**：与现有的持续集成流程集成

    ### 设置Git集成

    要将Modelbit与Git集成，你需要：

    1. **连接Git仓库**：
    ```python
    # 连接到GitHub仓库
    mb.connect_git(
        repo_url="https://github.com/your-username/your-repo.git",
        branch="main"
    )
    ```

    2. **配置部署分支**：
    ```python
    # 从特定分支部署
    mb.deploy(
        my_lr_deployment,
        branch="production",
        commit_sha="abc123def456"  # 可选：指定特定提交
    )
    ```

    ### 版本管理

    Modelbit自动为每个部署创建版本：

    ```python
    # 查看所有部署版本
    versions = mb.list_versions("my_lr_deployment")
    print(versions)

    # 部署特定版本
    mb.deploy(
        my_lr_deployment,
        version="v1.2.3"
    )

    # 回滚到之前版本
    mb.rollback("my_lr_deployment", version="v1.1.0")
    ```

    ### 分支策略

    推荐的分支策略：

    ```
    main/master     ← 稳定的生产代码
    ├── develop     ← 开发分支
    ├── feature/x   ← 功能开发分支
    └── hotfix/y    ← 紧急修复分支
    ```

    对应的部署策略：

    ```python
    # 开发环境部署
    mb.deploy(my_model, branch="develop", environment="dev")

    # 测试环境部署
    mb.deploy(my_model, branch="main", environment="staging")

    # 生产环境部署
    mb.deploy(my_model, branch="main", environment="production")
    ```

    ---

    ## 模型监控和日志记录

    部署模型后，监控其性能和行为至关重要。Modelbit提供了内置的监控和日志记录功能。

    ### 访问部署日志

    部署模型后，你可以在Modelbit仪表板中跟踪其使用情况并查看预测日志。

    选择你想要查看日志和分析使用情况的特定部署。

    你可以在"📚日志"部分查看日志，在"📊使用情况"部分查看模型的使用情况。

    ### 监控指标

    Modelbit自动跟踪以下指标：

    1. **性能指标**：
       - 请求延迟
       - 吞吐量（每秒请求数）
       - 错误率
       - 可用性

    2. **业务指标**：
       - 预测分布
       - 输入数据统计
       - 模型置信度
       - 用户行为模式

    3. **系统指标**：
       - CPU使用率
       - 内存使用率
       - 网络I/O
       - 存储使用

    ### 设置警报

    你可以为关键指标设置警报：

    ```python
    # 设置延迟警报
    mb.set_alert(
        deployment="my_lr_deployment",
        metric="latency",
        threshold=1000,  # 毫秒
        condition="greater_than"
    )

    # 设置错误率警报
    mb.set_alert(
        deployment="my_lr_deployment",
        metric="error_rate",
        threshold=0.05,  # 5%
        condition="greater_than"
    )
    ```

    ### 自定义日志记录

    你也可以在推理函数中添加自定义日志记录：

    ```python
    import logging

    def my_logged_deployment(input_x):
        \"\"\"
        带有自定义日志记录的推理函数
        \"\"\"
        # 记录输入
        logging.info(f"收到输入: {input_x}")

        try:
            # 预处理
            processed_input = preprocess(input_x)
            logging.info(f"预处理完成: {processed_input}")

            # 预测
            prediction = model.predict(processed_input)
            logging.info(f"预测结果: {prediction}")

            return prediction.tolist()

        except Exception as e:
            logging.error(f"预测失败: {str(e)}")
            raise
    ```

    ### 数据漂移检测

    Modelbit还可以帮助检测数据漂移：

    ```python
    # 配置数据漂移监控
    mb.configure_drift_detection(
        deployment="my_lr_deployment",
        reference_data=training_data,  # 参考数据集
        drift_threshold=0.1,           # 漂移阈值
        check_frequency="daily"        # 检查频率
        )
        ```

        当检测到数据漂移时，系统会自动发送警报，提醒你可能需要重新训练模型。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 高级功能

        除了基本的部署功能，Modelbit还提供了许多高级功能来支持企业级ML操作。

        ### 1. 云端训练

        除了在本地训练模型，我们还可以通过向Modelbit发送训练作业在云端训练模型，并直接在Modelbit的服务器上（如果需要的话，使用GPU）训练模型。

        ```python
        # 定义训练函数
        def train_model(data_path, hyperparameters):
        \"\"\"
        云端训练函数
        \"\"\"
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # 加载数据
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)
        y = data['target']

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        # 评估模型
        accuracy = model.score(X_test, y_test)

        return model, accuracy

    # 提交云端训练作业
    job = mb.train(
        train_model,
        data_path="s3://my-bucket/training-data.csv",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        compute_type="gpu",  # 使用GPU
        python_packages=["pandas", "scikit-learn"]
    )

    # 监控训练进度
    status = mb.get_job_status(job.id)
    print(f"训练状态: {status}")
    ```

    ### 2. 数据库集成

    通过连接到Snowflake或Amazon Redshift，Modelbit会自动为你部署的模型的每个版本创建SQL函数。我们可以直接在数据库中使用此SQL函数进行推理。

    ```python
    # 配置数据库连接
    mb.connect_database(
        type="snowflake",
        host="your-account.snowflakecomputing.com",
        database="ML_MODELS",
        schema="PREDICTIONS",
        credentials={
            "username": "your_username",
            "password": "your_password"
        }
    )

    # 部署后，可以在SQL中使用
    # SELECT
    #     customer_id,
    #     ML_MODELS.PREDICTIONS.my_lr_deployment(feature1, feature2) as prediction
    # FROM customer_data;
    ```

    ### 3. A/B测试

    Modelbit支持内置的A/B测试功能：

    ```python
    # 设置A/B测试
    mb.create_ab_test(
        name="model_comparison",
        control_deployment="my_lr_deployment_v1",
        treatment_deployment="my_lr_deployment_v2",
        traffic_split=0.5,  # 50/50分流
        success_metric="conversion_rate"
    )

    # 监控A/B测试结果
    results = mb.get_ab_test_results("model_comparison")
    print(f"控制组转化率: {results.control.conversion_rate}")
    print(f"实验组转化率: {results.treatment.conversion_rate}")
    print(f"统计显著性: {results.statistical_significance}")
    ```

    ### 4. 模型解释性

    集成模型解释工具：

    ```python
    import shap

    def explainable_deployment(input_x):
        \"\"\"
        带有解释性的推理函数
        \"\"\"
        # 进行预测
        prediction = model.predict([input_x])

        # 生成SHAP解释
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values([input_x])

        return {
            "prediction": prediction[0],
            "explanation": {
                "shap_values": shap_values[0].tolist(),
                "feature_names": feature_names,
                "base_value": explainer.expected_value
            }
        }

    # 部署可解释模型
    mb.deploy(
        explainable_deployment,
        python_packages=["shap", "scikit-learn"]
    )
    ```

    ### 5. 批量推理

    对于大规模数据处理：

    ```python
    # 定义批量推理函数
    def batch_inference(data_batch):
        \"\"\"
        批量推理函数
        \"\"\"
        import pandas as pd

        # 处理批量数据
        df = pd.DataFrame(data_batch)
        predictions = model.predict(df)

        # 返回结果
        return {
            "predictions": predictions.tolist(),
            "batch_size": len(data_batch),
            "processing_time": time.time()
        }

    # 部署批量推理
    mb.deploy(
        batch_inference,
        compute_type="cpu_large",  # 使用大型CPU实例
        timeout=300  # 5分钟超时
    )
    ```

    ---

    ## 最佳实践

    基于我们对Modelbit的探索，以下是一些最佳实践建议：

    ### 1. 开发流程

    ```python
    # 推荐的开发流程
    def recommended_workflow():
        # 1. 本地开发和测试
        model = train_model(training_data)
        test_locally(model, test_data)

        # 2. 版本控制
        git_commit("feat: 新的模型版本 v1.2.0")

        # 3. 部署到开发环境
        mb.deploy(model_function, environment="dev")

        # 4. 集成测试
        run_integration_tests()

        # 5. 部署到生产环境
        mb.deploy(model_function, environment="prod")

        # 6. 监控和警报
        setup_monitoring_alerts()
    ```

    ### 2. 错误处理

    ```python
    def robust_deployment(input_x):
        \"\"\"
        健壮的推理函数，包含完整的错误处理
        \"\"\"
        try:
            # 输入验证
            if not validate_input(input_x):
                return {"error": "无效输入", "code": 400}

            # 预处理
            processed_input = preprocess(input_x)

            # 预测
            prediction = model.predict(processed_input)

            # 后处理
            result = postprocess(prediction)

            return {
                "prediction": result,
                "confidence": get_confidence(prediction),
                "version": "v1.2.0",
                "timestamp": datetime.now().isoformat()
            }

        except ValidationError as e:
            logging.error(f"验证错误: {e}")
            return {"error": "输入验证失败", "code": 400}

        except ModelError as e:
            logging.error(f"模型错误: {e}")
            return {"error": "模型预测失败", "code": 500}

        except Exception as e:
            logging.error(f"未知错误: {e}")
            return {"error": "内部服务器错误", "code": 500}
    ```

    ### 3. 性能优化

    ```python
    # 性能优化技巧
    def optimized_deployment(input_x):
        \"\"\"
        优化的推理函数
        \"\"\"
        # 1. 缓存常用计算
        @lru_cache(maxsize=1000)
        def cached_preprocess(input_str):
            return expensive_preprocessing(input_str)

        # 2. 批量处理
        if isinstance(input_x, list) and len(input_x) > 1:
            return batch_predict(input_x)

        # 3. 早期返回
        if is_simple_case(input_x):
            return simple_prediction(input_x)

        # 4. 异步处理（如果适用）
        return complex_prediction(input_x)
    ```

    ### 4. 安全考虑

    ```python
    def secure_deployment(input_x, api_key=None):
        \"\"\"
        安全的推理函数
        \"\"\"
        # 1. API密钥验证
        if not validate_api_key(api_key):
            return {"error": "未授权", "code": 401}

        # 2. 输入清理
        sanitized_input = sanitize_input(input_x)

        # 3. 速率限制
        if is_rate_limited(api_key):
            return {"error": "请求过于频繁", "code": 429}

        # 4. 审计日志
        log_request(api_key, sanitized_input)

        # 进行预测
        prediction = model.predict(sanitized_input)

        return {"prediction": prediction}
        ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    通过这篇文章，我们深入探讨了机器学习模型部署的完整生命周期。

    在之前的文章中，我们详细研究了各种模型压缩技术如何有助于：

    1. 加速模型推理阶段
    2. 减少模型的大小

    但是，只有当我们打算为某些最终用户服务时，才会进行模型压缩。

    为最终用户服务只有在我们通过部署模型设置API端点时才可能实现。

    ![部署流程](https://www.dailydoseofds.com/content/images/2023/09/Screen-Recording-2023-09-22-at-12.55.29-PM.gif)

    特别是，我们讨论了如何使用Modelbit直接从Jupyter Notebook部署、版本控制和管理所有机器学习模型。

    在我看来，与其他高级部署服务（如AWS、Azure等）相比，Modelbit的学习曲线极其简单。

    当然，当托管高度可扩展的ML模型时，这些高级服务将是你的首选。

    然而，如果情况并非如此（这对大多数模型来说都是如此），我发现使用Modelbit部署模型极其简单和直接。

    👉

    Modelbit确实声称它可以托管相当大的模型，但我个人没有大规模测试过。

    当然，Modelbit可以做许多我们在本文中没有讨论的其他事情。

    ### 关键收获

    通过本文的学习，我们掌握了：

    #### 🎯 **部署后的关键考虑**
    - **版本控制**：跟踪模型变更，支持回滚和协作
    - **模型日志记录**：监控性能，检测漂移和异常
    - **持续监控**：确保模型在生产环境中的可靠性

    #### 🚀 **Modelbit的核心优势**
    - **一行代码部署**：极简的部署流程
    - **自动版本控制**：内置的模型版本管理
    - **Git集成**：与现有开发工作流程无缝集成
    - **内置监控**：实时性能和使用情况跟踪
    - **零基础设施管理**：无需配置服务器或容器

    #### 🛠️ **实践技能**
    - 从Jupyter Notebook直接部署ML模型
    - 定义和优化推理函数
    - 管理部署依赖和环境
    - 处理额外文件和通用资源
    - 设置监控和警报系统
    - 实施A/B测试和模型比较

    #### 📊 **高级功能**
    - **云端训练**：在Modelbit服务器上训练模型
    - **数据库集成**：直接在SQL中使用模型
    - **批量推理**：处理大规模数据
    - **模型解释性**：集成SHAP等解释工具
    - **安全和认证**：API密钥和访问控制

    ### 最佳实践总结

    1. **开发流程**：
       - 本地开发 → 版本控制 → 开发环境测试 → 生产部署
       - 完整的错误处理和输入验证
       - 性能优化和缓存策略

    2. **监控策略**：
       - 设置关键指标的警报阈值
       - 定期检查数据漂移
       - 维护详细的审计日志

    3. **安全考虑**：
       - API密钥验证和速率限制
       - 输入清理和验证
       - 访问控制和权限管理

    ### 未来展望

    随着机器学习在各行各业的广泛应用，模型部署和管理变得越来越重要。Modelbit这样的平台通过简化部署流程，让数据科学家能够专注于模型开发而不是基础设施管理。

    未来的发展方向可能包括：

    - **更智能的自动化**：自动模型重训练和部署
    - **更强的可观察性**：深度模型行为分析
    - **更好的集成**：与更多数据平台和工具的集成
    - **边缘部署**：支持移动设备和IoT设备部署

    ### 行动建议

    如果你是数据科学家或ML工程师，建议你：

    1. **动手实践**：使用Modelbit部署一个简单的模型
    2. **建立流程**：为你的团队建立标准化的部署流程
    3. **监控优先**：从第一天就设置监控和警报
    4. **持续学习**：跟上MLOps领域的最新发展

    记住，成功的机器学习项目不仅仅是训练一个好模型，更重要的是能够可靠地为用户提供价值。而这正是优秀的部署和管理实践所能带来的。

    ---

    ## 参考资源

    - [Modelbit官方文档](https://docs.modelbit.com/)
    - [MLOps最佳实践指南](https://ml-ops.org/)
    - [模型监控和可观察性](https://neptune.ai/blog/ml-model-monitoring-best-tools)
    - [机器学习系统设计](https://huyenchip.com/machine-learning-systems-design/)

    💡 **记住**：部署只是开始，持续的监控、维护和改进才是确保ML系统长期成功的关键！
    """
    )
    return


if __name__ == "__main__":
    app.run()

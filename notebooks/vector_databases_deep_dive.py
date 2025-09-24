import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 向量数据库的初学者友好且全面的深度解析

    **发布日期：** 2024年2月18日

    **作者：** Avi Chawla

    理解向量数据库及其在大语言模型中的实用性的每一个细节，并附有实践演示。

    ![作者头像](https://www.dailydoseofds.com/content/images/size/w100/2024/12/avi-google.jpg)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 介绍

    在生成式AI时代（更准确地说，自ChatGPT发布以来），你很可能至少听说过"**向量数据库**"这个术语。

    如果你不知道它们是什么，这完全没关系，因为这篇文章主要是为了详细解释关于向量数据库的一切。

    但考虑到它们最近变得如此流行，我认为了解是什么让它们如此强大以至于获得如此多的关注，以及它们不仅在大语言模型中而且在其他应用中的实际用途是至关重要的。

    让我们深入了解吧！
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 什么是向量数据库？

    ### 目标

    首先，我们必须注意到向量数据库并不是新的。

    事实上，它们已经存在了相当长的时间。即使在它们最近广泛流行之前，你也一直在间接地与它们互动。这些应用包括推荐系统和搜索引擎等。

    简单来说，向量数据库以**向量嵌入**的形式存储**非结构化数据**（文本、图像、音频、视频等）。

    ![向量数据库概念图](https://www.dailydoseofds.com/content/images/2024/02/image-160.png)

    每个数据点，无论是单词、文档、图像还是任何其他实体，都使用机器学习技术转换为数值向量（我们将在后面看到）。

    这个数值向量被称为**嵌入**，模型经过训练，使这些向量能够捕获底层数据的基本特征和特性。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    考虑词嵌入，例如，我们可能会发现在嵌入空间中，水果的嵌入彼此靠近，城市形成另一个聚类，等等。

    ![词嵌入聚类](https://www.dailydoseofds.com/content/images/2024/02/image-161.png)

    这表明嵌入可以学习它们所代表实体的语义特征（前提是它们经过适当训练）。

    一旦存储在向量数据库中，我们就可以检索与我们希望在非结构化数据上运行的查询相似的原始对象。

    ![向量检索过程](https://www.dailydoseofds.com/content/images/2024/02/image-162.png)

    换句话说，编码**非结构化数据**允许我们对其运行许多复杂的操作，如相似性搜索、聚类和分类，这在传统数据库中是困难的。

    举例来说，当电子商务网站为类似商品提供推荐或基于输入查询搜索产品时，我们（**在大多数情况下**）在幕后与向量数据库交互。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    在我们深入技术细节之前，让我给你几个直观的例子来理解向量数据库及其巨大的实用性。

    ### 示例 #1

    让我们想象一下，我们有一个多年来各种假期拍摄的照片集合。每张照片都捕捉了不同的场景，如海滩、山脉、城市和森林。

    ![照片集合](https://www.dailydoseofds.com/content/images/2024/02/image-163.png)

    现在，我们想要以一种更容易快速找到相似照片的方式来组织这些照片。

    传统上，我们可能会按拍摄日期或拍摄地点来组织它们。

    ![传统组织方式](https://www.dailydoseofds.com/content/images/2024/02/image-164.png)

    然而，我们可以通过将它们编码为向量来采用更复杂的方法。

    更具体地说，我们可以将每张照片表示为一组捕获图像本质的数值向量，而不是仅仅依赖日期或位置。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    💡 **注意：** 虽然Google Photos没有明确披露其后端系统的确切技术细节，但我推测它使用向量数据库来促进其图像搜索和组织功能，你可能已经多次使用过。

    假设我们使用一种算法，根据每张照片的颜色组成、突出形状、纹理、人物等将其转换为向量。

    现在每张照片都被表示为多维空间中的一个点，其中维度对应于图像中的不同视觉特征和元素。

    现在，当我们想要找到相似的照片时，比如基于我们的输入文本查询，我们将文本查询编码为向量并与图像向量进行比较。

    匹配查询的照片预期在这个多维空间中具有彼此接近的向量。

    假设我们希望找到山脉的图像。

    在这种情况下，我们可以通过查询向量数据库中接近表示输入查询的向量的图像来快速找到这样的照片。

    ![山脉图像搜索](https://www.dailydoseofds.com/content/images/2024/02/image-165.png)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    这里需要注意的一点是，向量数据库不仅仅是一个跟踪嵌入的数据库。

    相反，它既维护嵌入又维护生成这些嵌入的原始数据。

    ![向量数据库结构](https://www.dailydoseofds.com/content/images/2024/02/image-166.png)

    你可能想知道，为什么这是必要的？

    再次考虑上面的图像检索任务，如果我们的向量数据库只由向量组成，我们还需要一种重建图像的方法，因为这是最终用户需要的。

    当用户查询山脉图像时，他们会收到表示相似图像的向量列表，但没有实际图像。

    ![缺少原始数据的问题](https://www.dailydoseofds.com/content/images/2024/02/image-167.png)

    通过存储嵌入（表示图像的向量）和原始图像数据，向量数据库确保当用户查询相似图像时，它不仅返回最接近的匹配向量，还提供对原始图像的访问。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 示例 #2

    在这个例子中，考虑一个全文本非结构化数据，比如数千篇新闻文章，我们希望从该数据中搜索答案。

    ![新闻文章搜索](https://www.dailydoseofds.com/content/images/2024/02/image-169.png)

    传统的搜索方法依赖于精确的关键词搜索，这完全是一种暴力方法，不考虑文本数据的固有复杂性。

    换句话说，语言是极其细致入微的，每种语言都提供了表达同一想法或提出同一问题的各种方式。

    例如，像"今天天气怎么样？"这样的简单询问可以用许多方式表达，如"今天天气如何？"、"外面阳光明媚吗？"或"当前的天气条件如何？"。

    这种语言多样性使传统的基于关键词的搜索方法变得不足。

    正如你可能已经猜到的，在这种情况下，将这些数据表示为向量也可能非常有用。

    我们可以首先在高维向量空间中表示文本数据并将它们存储在向量数据库中，而不是仅仅依赖关键词并遵循暴力搜索。

    ![文本向量化](https://www.dailydoseofds.com/content/images/2024/02/image-170.png)

    当用户提出查询时，向量数据库可以将查询的向量表示与文本数据的向量表示进行比较，**即使它们不共享完全相同的措辞。**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 如何生成嵌入？

    此时，如果你想知道我们如何将单词（字符串）转换为向量（数字列表），让我解释一下。

    我们在最近的一期新闻通讯中也涵盖了这一点，但没有太多细节，所以让我们在这里讨论这些细节。

    ![嵌入模型图标](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a039798-62e1-4323-b47f-4fa3c6bf6744%2Fapple-touch-icon-180x180.png)

    > 如果你已经知道什么是嵌入模型，请随意跳过这一部分。

    要构建面向语言任务的模型，为单词生成数值表示（或向量）是至关重要的。

    这允许单词被数学处理和操作，并对单词执行各种计算操作。

    嵌入的目标是捕获单词之间的语义和句法关系。这有助于机器更有效地理解和推理语言。

    在Transformer时代之前，这主要使用预训练的静态嵌入来完成。

    本质上，有人会使用深度学习技术在比如100k或200k常见单词上训练嵌入并开源它们。

    ![静态嵌入](https://www.dailydoseofds.com/content/images/2024/02/image-171.png)

    因此，其他研究人员会在他们的项目中利用这些嵌入。

    当时（大约2013-2017年）最流行的模型是：

    - Glove
    - Word2Vec
    - FastText等

    这些嵌入在学习单词之间的关系方面确实显示了一些有希望的结果。

    例如，当时的一个实验表明，向量运算 `(King - Man) + Woman` 返回了一个接近单词 `Queen` 的向量。

    ![词向量运算](https://www.dailydoseofds.com/content/images/2024/02/image-173.png)

    这很有趣，不是吗？

    事实上，以下关系也被发现是真实的：

    - `Paris - France + Italy` ≈ `Rome`
    - `Summer - Hot + Cold` ≈ `Winter`
    - `Actor - Man + Woman` ≈ `Actress`
    - 等等。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    所以，虽然这些嵌入捕获了相对的词表示，但有一个主要限制。

    考虑以下两个句子：

    - 将此数据转换为Excel中的**表格**。
    - 将此瓶子放在**桌子**上。

    这里，单词"**table**"传达了两种完全不同的含义：

    - 第一个句子指的是单词"table"的"**数据**"特定含义。
    - 第二个句子指的是单词"table"的"**家具**"特定含义。

    然而，静态嵌入模型为它们分配了相同的表示。

    ![静态嵌入的限制](https://www.dailydoseofds.com/content/images/2024/02/image-174.png)

    因此，这些嵌入没有考虑到一个单词在不同上下文中可能有不同的用法。

    但这在Transformer时代得到了解决，这导致了由Transformer驱动的上下文化嵌入模型，例如：

    **BERT**: 使用两种技术训练的语言模型：

    - 掩码语言建模（MLM）：在给定周围单词的情况下预测句子中的缺失单词。
    - 下一句预测（NSP）。
    - *我们将很快更详细地讨论它。*

    **DistilBERT**: BERT的简单、有效且更轻量的版本，大约小40%：

    - 利用称为师生理论的常见机器学习策略。
    - 这里，学生是BERT的蒸馏版本，老师是原始BERT模型。
    - 学生模型应该复制老师模型的行为。

    ![DistilBERT图标](https://www.dailydoseofds.com/content/images/size/w256h256/format/png/2023/06/logo-subsatck2-1.svg)

    **SentenceTransformer**: 如果你阅读了最近关于在序数数据上构建分类模型的深度解析，我们在那里讨论了这个模型。

    - 本质上，**SentenceTransformer**模型接受整个句子并为该句子生成嵌入。

    ![SentenceTransformer](https://www.dailydoseofds.com/content/images/2024/02/image-175.png)

    - 这与BERT和DistilBERT模型不同，后者为句子中的所有单词产生嵌入。

    还有更多模型，但我们不会在这里详细介绍，我希望你明白这一点。

    这个想法是，由于它们的自注意力机制和适当的训练机制，这些模型非常能够生成上下文感知的表示。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### BERT

    例如，如果我们再次考虑BERT，我们上面讨论过它使用掩码语言建模（MLM）技术和下一句预测（NSP）。

    这些步骤也被称为BERT的**预训练步骤**，因为它们涉及在针对特定下游任务进行微调之前在大型文本数据语料库上训练模型。

    💡 **预训练**，在机器学习模型训练的背景下，指的是训练的初始阶段，其中模型从大型文本数据语料库中学习一般语言表示。预训练的目标是使模型能够捕获语言的句法和语义属性，如语法、上下文和单词之间的关系。虽然文本本身是未标记的，但MLM和NSP是两个帮助我们以监督方式训练模型的任务。一旦模型被训练，我们就可以使用模型从预训练阶段获得的语言理解能力，并在特定任务数据上微调模型。以下动画描述了微调：

    ![微调动画](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9764beac-a786-4305-9a47-ec050b0ebef6_1060x308.gif)

    继续，让我们看看掩码语言建模（MLM）和下一句预测（NSP）的预训练目标如何帮助BERT生成嵌入。

    #### **#1) 掩码语言建模（MLM）**

    - 在MLM中，BERT被训练来预测句子中的缺失单词。为了做到这一点，**大多数**（不是全部）句子中的一定百分比的单词被随机替换为特殊标记`[MASK]`。

    ![MLM过程](https://www.dailydoseofds.com/content/images/2024/02/image-176.png)

    - 然后BERT双向处理掩码句子，这意味着它考虑每个掩码单词的左右上下文，这就是为什么名称是"**双向**编码器表示来自Transformers（BERT）"。

    ![双向处理](https://www.dailydoseofds.com/content/images/2024/02/image-177.png)

    - 对于每个掩码单词，BERT从其上下文预测原始单词应该是什么。它通过在整个词汇表上分配概率分布并选择具有最高概率的单词作为预测单词来做到这一点。

    ![预测过程](https://www.dailydoseofds.com/content/images/2024/02/image-179.png)

    - 在训练期间，BERT被优化以最小化预测单词和实际掩码单词之间的差异，使用交叉熵损失等技术。

    #### **#2) 下一句预测（NSP）**

    - 在NSP中，BERT被训练来确定两个输入句子是否在文档中连续出现，或者它们是否是来自不同文档的随机配对句子。

    ![NSP过程](https://www.dailydoseofds.com/content/images/2024/02/image-180.png)

    - 在训练期间，BERT接收句子对作为输入。这些对中的一半是来自同一文档的连续句子（正例），另一半是来自不同文档的随机配对句子（负例）。

    ![NSP训练数据](https://www.dailydoseofds.com/content/images/2024/02/image-182.png)

    - 然后BERT学习预测第二个句子是否跟随原始文档中的第一个句子（`标签1`）或者它是否是随机配对的句子（`标签0`）。
    - 与MLM类似，BERT被优化以最小化预测标签和实际标签之间的差异，使用二元交叉熵损失等技术。

    💡 如果我们回顾MLM和NSP，在这两种情况下，我们一开始都不需要标记的数据集。相反，我们使用文本本身的结构来创建训练示例。这允许我们利用大量未标记的文本数据，这通常比标记数据更容易获得。

    现在，让我们看看这些预训练目标如何帮助BERT生成嵌入：

    - **MLM：** 通过基于上下文预测掩码单词，BERT学习捕获句子中每个单词的含义和上下文。BERT生成的嵌入不仅反映单词的个别含义，还反映它们与句子中周围单词的关系。
    - **NSP：** 通过确定句子是否连续，BERT学习理解文档中不同句子之间的关系。这有助于BERT生成不仅捕获个别句子含义而且捕获文档或文本段落更广泛上下文的嵌入。

    通过一致的训练，模型学习不同单词在句子中如何相互关联。它学习哪些单词经常一起出现以及它们如何适应句子的整体含义。

    这个学习过程帮助BERT为单词和句子创建**上下文化**的嵌入，这与早期的嵌入如Glove和Word2Vec不同：

    ![上下文化嵌入](https://www.dailydoseofds.com/content/images/2024/02/image-183.png)

    上下文化意味着嵌入模型可以基于单词使用的上下文动态生成单词的嵌入。

    因此，如果一个单词出现在不同的上下文中，模型会返回不同的表示。

    这在下面的图像中精确描述了单词`Bank`的不同用法。

    > 为了可视化目的，嵌入已使用t-SNE投影到2d空间。

    ![Bank词的不同含义](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe3ea13e1-2e9b-4030-955f-85751d9fca97_2454x2439.jpeg)

    如上所示，静态嵌入模型——Glove和Word2Vec为单词的不同用法产生相同的嵌入。

    然而，上下文化嵌入模型不会。

    事实上，上下文化嵌入理解单词"Bank"的不同含义/意义：

    - 金融机构
    - 倾斜的土地
    - 长脊等等。

    ![Bank的不同含义](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5462f667-fb98-423e-887f-fee3f54533e6_2533x931.png)

    因此，这些上下文化嵌入模型解决了静态嵌入模型的主要限制。

    上述讨论的要点是现代嵌入模型在编码任务方面非常熟练。

    因此，它们可以轻松地将文档、段落或句子转换为捕获其语义含义和上下文的数值向量。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 查询向量数据库

    在倒数第二个子部分中，我们提供了一个输入查询，该查询被编码，然后我们在向量数据库中搜索与输入向量**相似**的向量。

    ![查询过程](https://www.dailydoseofds.com/content/images/2024/02/image-167.png)

    换句话说，目标是返回由相似性度量测量的**最近邻**，这可能是：

    - 欧几里得距离（度量越低，相似性越高）。
    - 曼哈顿距离（度量越低，相似性越高）。
    - 余弦相似性（度量越高，相似性越高）。

    这个想法与我们在典型的k最近邻（kNN）设置中所做的相呼应。

    ![kNN概念](https://www.dailydoseofds.com/content/images/2024/02/image-184.png)

    我们可以将查询向量与已编码的向量进行匹配，并返回最相似的向量。

    这种方法的问题是，要找到比如只是第一个最近邻，输入查询必须与向量数据库中存储的**所有**向量进行匹配。

    ![暴力搜索](https://www.dailydoseofds.com/content/images/2024/02/image-185.png)

    这在计算上是昂贵的，特别是在处理可能有数百万数据点的大型数据集时。随着向量数据库大小的增长，执行最近邻搜索所需的时间成比例增加。

    ![性能问题](https://www.dailydoseofds.com/content/images/2024/02/image-186.png)

    但在需要实时或近实时响应的场景中，这种暴力方法变得不切实际。

    事实上，这个问题在典型的关系数据库中也观察到。如果我们要获取匹配特定条件的行，必须扫描整个表。

    ![数据库扫描](https://www.dailydoseofds.com/content/images/2024/02/ezgif.com-animated-gif-maker.gif)

    索引数据库提供了快速查找机制，特别是在近实时延迟至关重要的情况下。

    更具体地说，当在`WHERE`子句或`JOIN`条件中使用的列被索引时，它可以显著加快查询性能。

    向量数据库中也使用了类似的**索引**想法，这导致了我们称之为**近似最近邻（ANN）**的东西，这是相当自解释的，不是吗？

    嗯，核心思想是在准确性和运行时间之间进行权衡。因此，近似最近邻算法用于找到数据点的最近邻，尽管这些邻居可能并不总是最近的邻居。

    这就是为什么它们也被称为**非穷尽**搜索算法。

    动机是当我们使用向量搜索时，在大多数情况下精确匹配并不是绝对必要的。

    近似最近邻（ANN）算法利用这一观察，为运行时效率牺牲一点准确性。

    因此，ANN算法不是穷尽搜索数据库中的所有向量以找到最接近的匹配，而是提供快速的、亚线性时间复杂度的解决方案，产生近似最近邻。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **近似最近邻（ANN）**

    虽然近似最近邻算法与精确最近邻方法相比可能会牺牲一定程度的精度，但它们提供了显著的性能提升，特别是在需要实时或近实时响应的场景中。

    **核心思想是缩小查询向量的搜索空间，从而提高运行时性能。**

    ![搜索空间缩小](https://www.dailydoseofds.com/content/images/2024/02/image-187.png)

    搜索空间通过**索引**的帮助得到缩小。这里有五种流行的索引策略。

    让我们逐一了解它们。

    #### #1) 平面索引

    平面索引是我们之前看到的暴力搜索的另一个名称，这也是KNN所做的。因此，所有向量都存储在单个索引结构中，没有任何分层组织。

    ![平面索引](https://www.dailydoseofds.com/content/images/2024/02/image-188.png)

    这就是为什么这种索引技术被称为"平面"——它不涉及索引策略，并按原样存储数据向量，即在"平面"数据结构中。

    ![平面数据结构](https://www.dailydoseofds.com/content/images/2024/02/image-189.png)

    由于它搜索整个向量数据库，它在我们将看到的所有索引方法中提供最佳准确性。然而，这种方法非常慢且不切实际。

    尽管如此，当数据条件有利时，比如只有少数数据点要搜索和低维数据集，我不会建议采用任何其他复杂方法而不是平面索引。

    但是，当然，并非所有数据集都很小，在大多数现实生活情况下使用平面索引是不切实际的。

    因此，我们需要更复杂的方法来索引向量数据库中的向量。

    #### #2) 倒排文件索引

    IVF可能是最简单和最直观的索引技术之一。虽然它通常用于文本检索系统，但它可以适应向量数据库进行近似最近邻搜索。

    这是如何做的！

    给定高维空间中的一组向量，想法是将它们组织成不同的分区，通常使用k-means等聚类算法。

    ![IVF分区](https://www.dailydoseofds.com/content/images/2024/02/image-190.png)

    因此，每个分区都有相应的质心，每个向量都与对应于其最近质心的**仅一个**分区相关联。

    ![质心关联](https://www.dailydoseofds.com/content/images/2024/02/image-191.png)

    因此，每个质心维护关于属于其分区的所有向量的信息。

    ![质心信息](https://www.dailydoseofds.com/content/images/2024/02/image-193.png)

    当搜索查询向量的最近向量时，我们不是搜索所有向量，而是首先找到查询向量的最近**质心**。

    ![找到最近质心](https://www.dailydoseofds.com/content/images/2024/02/image-194.png)

    然后在仅属于上面找到的最近质心分区的那些向量中搜索最近邻。

    ![分区内搜索](https://www.dailydoseofds.com/content/images/2024/02/image-198.png)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    让我们估计它相对于使用平面索引提供的搜索运行时差异。

    重申一下，在平面索引中，我们计算查询向量与向量数据库中所有向量的相似性。

    ![平面索引复杂度](https://www.dailydoseofds.com/content/images/2024/02/image-199.png)

    如果我们有N个向量，每个向量是D维的，运行时复杂度是O(ND)来找到最近向量。

    将其与倒排文件索引进行比较，其中我们首先计算查询向量与使用聚类算法获得的所有**质心**的相似性。

    ![IVF复杂度](https://www.dailydoseofds.com/content/images/2024/02/image-202.png)

    假设有k个质心，总共N个向量，每个向量是D维的。

    另外，为了简单起见，假设向量在所有分区中均匀分布。因此，每个分区将有N/k个数据点。

    首先，我们计算查询向量与所有**质心**的相似性，其运行时复杂度是O(kD)。

    ![质心比较复杂度](https://www.dailydoseofds.com/content/images/2024/02/image-203.png)

    接下来，我们计算查询向量与属于质心分区的数据点的相似性，运行时复杂度为O(ND/k)。

    ![分区比较复杂度](https://www.dailydoseofds.com/content/images/2024/02/image-204.png)

    因此，总体运行时复杂度为O(kD + ND/k)。

    ![总体复杂度](https://www.dailydoseofds.com/content/images/2024/02/image-296.png)

    为了获得一些视角，假设我们在向量数据库中有1000万个向量，并将其分为k=100个质心。因此，每个分区预期大约有10万个数据点。

    在平面索引中，我们将在所有数据点——**1000万**中比较输入查询。

    在IVF中，首先，我们将在所有质心（`100`）中比较输入查询，然后将其与获得的分区中的向量（`10万`）进行比较。因此，在这种情况下，总比较次数将是`100,050`，**几乎快`100`倍**。

    当然，重要的是要注意，如果一些向量实际上接近输入向量但仍然恰好在相邻分区中，我们将错过它们。

    ![邻近分区问题](https://www.dailydoseofds.com/content/images/2024/02/image-206.png)

    但回想我们的目标，我们从来没有瞄准最佳解决方案，而是近似最佳（这就是为什么我们称之为"近似最近邻"），所以这种准确性权衡是我们为了更好的运行时性能而愿意接受的。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### #3) 产品量化

    量化的一般思想是指在保留原始信息的同时压缩数据。

    因此，产品量化（PQ）是一种用于内存高效最近邻搜索的向量压缩技术。

    让我们详细了解它是如何工作的。

    ##### 步骤1）创建数据段

    假设我们有一些向量，每个向量是`256`维的。假设每个维度由占用`32`位的数字表示，每个向量消耗的内存将是`256` x `32`位 = `8192`位。

    ![向量内存消耗](https://www.dailydoseofds.com/content/images/2024/02/image-208.png)

    在产品量化（PQ）中，我们首先将所有向量分割成子向量。下面显示了一个演示，我们将向量分割成`M`（一个参数）段，比如`8`：

    ![向量分段](https://www.dailydoseofds.com/content/images/2024/02/image-209.png)

    因此，每个段将是`32`维的。

    ##### 步骤2）运行KMeans

    接下来，我们在每个段上分别运行KMeans，这将为每个段生成`k`个质心。

    ![KMeans聚类](https://www.dailydoseofds.com/content/images/2024/02/image-210.png)

    请注意，每个质心将代表子空间（`32`维）的质心，而不是整个向量空间（在此演示中为`256`维）。

    例如，如果`k=100`，这将总共生成`100*8`个质心。

    训练完成。

    ##### 步骤3）编码向量

    接下来，我们转到编码步骤。

    想法是对于整个数据库中向量的每个段，我们从相应段中找到最近的质心，该段具有在上面训练步骤中获得的`k`个质心。

    例如，考虑我们开始的`256`维数据的第一段：

    ![第一段编码](https://www.dailydoseofds.com/content/images/2024/02/image-211.png)
    n表示数据库中向量的数量

    我们将这些段向量与相应的`k`个质心进行比较，并为所有段向量找到最近的质心：

    ![找到最近质心](https://www.dailydoseofds.com/content/images/2024/02/image-212.png)

    在为每个向量段获得最近质心后，我们用唯一的`质心ID`替换整个段，这可以被认为是该子空间中质心的索引（从`0`到`k-1`的数字）。

    ![质心ID替换](https://www.dailydoseofds.com/content/images/2024/02/image-213.png)

    我们对向量的所有单独段都这样做：

    ![所有段编码](https://www.dailydoseofds.com/content/images/2024/02/image-214.png)

    这为我们提供了向量数据库中所有向量的量化（或压缩）表示，该表示由`质心ID`组成，它们也被称为**PQ码**。

    回想一下，我们在这里所做的是，我们已经用`质心ID`的向量编码了向量数据库中的所有向量，这是一个从`0`到`k-1`的数字，**每个维度现在只消耗8位内存**。

    由于有`8`个段，总内存消耗是`8*8=64`位，这比我们之前的内存使用量——`8192`位低128倍。

    当我们处理数百万个向量时，内存节省规模非常好。

    当然，编码表示并不完全准确，但不要担心，因为我们在所有方面都不需要完全精确。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### 步骤4）找到近似最近邻

    现在，你可能想知道，我们如何根据编码表示搜索最近邻？

    ![搜索过程](https://www.dailydoseofds.com/content/images/2024/02/image-215.png)

    更具体地说，给定一个新的查询向量`Q`，我们必须在我们的数据库中找到与`Q`最相似（或最接近）的向量。

    我们首先将查询向量`Q`分割成`M`段，就像我们之前做的那样。

    ![查询向量分段](https://www.dailydoseofds.com/content/images/2024/02/image-216.png)

    接下来，我们计算向量`Q`的所有段与从上面KMeans步骤获得的该段的所有相应质心之间的距离。

    ![距离计算](https://www.dailydoseofds.com/content/images/2024/02/image-217.png)

    这给我们一个距离矩阵：

    ![距离矩阵](https://www.dailydoseofds.com/content/images/2024/02/image-218.png)

    最后一步是估计查询向量`Q`与向量数据库中向量的距离。

    为了做到这一点，我们回到我们之前生成的PQ矩阵：

    ![PQ矩阵](https://www.dailydoseofds.com/content/images/2024/02/image-219.png)

    接下来，我们查找上面生成的距离矩阵中的相应条目。

    例如，上面PQ矩阵中的第一个向量是这样的：

    ![第一个向量](https://www.dailydoseofds.com/content/images/2024/02/image-220.png)

    要获得我们的查询向量到这个向量的距离，我们检查距离条目中的相应条目。

    ![距离查找](https://www.dailydoseofds.com/content/images/2024/02/image-221.png)

    我们将所有段级距离相加，以获得查询向量`Q`与向量数据库中所有向量距离的粗略估计。

    我们对数据库中的所有向量重复此操作，找到最低距离，并从数据库返回相应的向量。

    当然，重要的是要注意，上面的PQ矩阵查找仍然是暴力搜索。这是因为我们为PQ矩阵的所有条目查找距离矩阵中的所有距离。

    ![暴力搜索问题](https://www.dailydoseofds.com/content/images/2024/02/image-222.png)

    此外，由于我们不是估计向量到向量的距离，而是向量到质心的距离，获得的值只是近似距离而不是真实距离。

    增加质心和段的数量将增加近似最近邻搜索的精度，但也会增加搜索算法的运行时间。

    **以下是产品量化方法的总结：**

    - 将向量数据库中的向量分为M段。
    - 在每个段上运行KMeans。这将为每个段提供`k`个质心。
    - 通过用向量所属聚类的`质心ID`替换向量的每个段来编码向量数据库中的向量。这生成了一个PQ矩阵，它在内存方面非常高效。
    - 接下来，要确定查询向量`Q`的近似最近邻，生成一个距离矩阵，其每个条目表示向量`Q`的段到所有质心的**距离**。
    - 现在回到PQ码，并查找上面距离矩阵中的距离，以获得数据库中所有向量与查询向量`Q`之间距离的估计。选择具有最小距离的向量以获得近似最近邻。

    完成！

    使用产品量化的近似最近邻搜索适用于中等规模的系统，很明显精度和内存利用率之间存在权衡。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### #4-5) 分层可导航小世界（HNSW）

    HNSW可能是专门为高维空间中最近邻搜索设计的最有效和高效的索引方法之一。

    核心思想是构建一个图结构，其中每个节点代表一个数据向量，边根据它们的相似性连接节点。

    ![HNSW图结构](https://www.dailydoseofds.com/content/images/2024/02/image-224.png)

    HNSW以促进快速搜索操作的方式组织图，通过有效地导航图来找到近似最近邻。

    但在我们理解HNSW之前，理解**NSW（可导航小世界）**是至关重要的，它是HNSW算法的基础。

    即将到来的讨论基于你对图有一些了解的假设。

    虽然我们不能详细涵盖它们，但这里有一些细节足以理解即将到来的概念。

    图由顶点和边组成，其中边将顶点连接在一起。在这种情况下，连接的顶点通常被称为邻居。

    ![图的基本概念](https://www.dailydoseofds.com/content/images/2024/02/image-225.png)

    回想我们之前讨论的关于向量的内容，我们知道相似的向量通常在向量空间中彼此靠近。

    因此，如果我们将这些向量表示为图的顶点，彼此接近的顶点（即具有高相似性的向量）应该作为邻居连接。

    也就是说，即使两个节点没有直接连接，它们也应该可以通过遍历其他顶点到达。

    **这意味着我们必须创建一个可导航的图。**

    更正式地说，要使图可导航，每个顶点都必须有邻居；否则，将无法到达某些顶点。

    ![不可导航图](https://www.dailydoseofds.com/content/images/2024/02/image-226.png)

    另外，虽然有邻居对遍历有益，但同时，我们希望避免每个节点都有太多邻居的情况。

    ![过度连接图](https://www.dailydoseofds.com/content/images/2024/02/image-227.png)

    这在内存、存储和搜索时间的计算复杂性方面可能是昂贵的。

    理想情况下，我们希望一个可导航的图类似于小世界网络，其中每个顶点只有有限数量的连接，两个随机选择的顶点之间的平均边遍历数量很低。

    这种类型的图对于大型数据集中的相似性搜索是高效的。

    如果这很清楚，我们可以理解可导航小世界（NSW）算法是如何工作的。

    ##### → NSW中的图构建

    NSW的第一步是图构建，我们称之为`G`。

    这是通过随机打乱向量并通过以**随机顺序**顺序插入顶点来构建图来完成的。

    当向图（`G`）添加新顶点（`V`）时，它与图中最接近它的`K`个现有顶点共享边。

    这个演示会让它更清楚。

    假设我们设置`K=3`。

    最初，我们插入第一个顶点`A`。由于此时图中没有其他顶点，`A`保持未连接。

    ![插入第一个顶点](https://www.dailydoseofds.com/content/images/2024/02/image-228.png)

    接下来，我们添加顶点`B`，将其连接到`A`，因为`A`是唯一现有的顶点，它无论如何都会在前`K`个最近顶点中。现在图有两个顶点`{A, B}`。

    ![插入第二个顶点](https://www.dailydoseofds.com/content/images/2024/02/image-229.png)

    接下来，当插入顶点`C`时，它连接到`A`和`B`。顶点`D`也发生完全相同的过程。

    ![插入更多顶点](https://www.dailydoseofds.com/content/images/2024/02/image-232.png)

    现在，当顶点`E`插入图中时，它只连接到`K=3`个最近的顶点，在这种情况下是`A`、`B`和`D`。

    ![插入顶点E](https://www.dailydoseofds.com/content/images/2024/02/image-233.png)

    这个顺序插入过程继续，逐渐构建NSW图。

    好的是，随着越来越多的顶点被添加，在图构建的早期阶段形成的连接可能成为更长距离的链接，这使得在小跳跃中导航长距离变得更容易。

    这从以下图中很明显，其中连接`A — C`和`B — D`跨越更大的距离。

    ![长距离连接](https://www.dailydoseofds.com/content/images/2024/02/image-234.png)

    通过以这种方式构建图，我们得到一个NSW图，最重要的是，它是可导航的。

    **换句话说，图中的任何节点都可以在一些跳跃中从任何其他节点到达。**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### → NSW中的搜索

    在上面构建的NSW图（`G`）中，搜索过程使用简单的贪婪搜索方法进行，该方法在每一步都依赖于局部信息。

    假设我们想要在下面的图中找到黄色节点的最近邻：

    ![搜索目标](https://www.dailydoseofds.com/content/images/2024/02/image-235.png)

    要开始搜索，随机选择一个入口点，这也是该算法的美妙之处。换句话说，NSW的一个关键优势是可以从图`G`中的任何顶点开始搜索。

    让我们选择节点`A`作为入口点：

    ![选择入口点](https://www.dailydoseofds.com/content/images/2024/02/image-236.png)

    选择初始点后，算法迭代地找到最接近查询向量`Q`的邻居（即连接的顶点）。

    例如，在这种情况下，顶点`A`有邻居（`D`、`B`、`C`和`E`）。因此，我们将计算这`4`个邻居与查询向量`Q`的距离（或相似性，无论你选择什么作为度量）。

    在这种情况下，节点`C`是最近的，所以我们从节点`A`移动到节点`C`。

    ![移动到节点C](https://www.dailydoseofds.com/content/images/2024/02/image-238.png)

    接下来，搜索移向与查询向量距离最小的顶点。

    节点`C`的**未评估**邻居只有`H`，它恰好更接近查询向量，所以我们现在移动到节点`H`。

    ![移动到节点H](https://www.dailydoseofds.com/content/images/2024/02/image-239.png)

    重复此过程，直到找不到更接近查询向量的邻居，这给我们图中查询向量的最近邻。

    我喜欢这个搜索算法的一点是它多么直观和易于实现。

    **也就是说，搜索仍然是近似的，不能保证我们总是找到最近的邻居，它可能返回高度次优的结果。**

    例如，考虑下面的图，其中节点`A`是入口点，黄色节点是我们需要最近邻的向量：

    ![次优结果示例](https://www.dailydoseofds.com/content/images/2024/02/image-241.png)

    按照上述最近邻搜索程序，我们将评估节点`A`的邻居，即`C`和`B`。

    很明显，两个节点都比节点`A`距离查询向量更远。因此，算法返回节点`A`作为最终最近邻。

    为了避免这种情况，建议使用多个入口点重复搜索过程，这当然会消耗更多时间。

    ##### → 跳跃列表数据结构

    虽然NSW是一个相当有前途和直观的方法，但另一个主要问题是我们最终会多次遍历图（或多次重复搜索）以到达最优的近似最近邻节点。

    HNSW通过将向量数据库索引到更优的图结构中来加速搜索过程，该结构基于**跳跃列表**的思想。

    首先，让我给你一些关于跳跃列表数据结构的细节，因为这在这里很重要。

    为了做到这一点，让我们考虑一个非常直观的例子，这将使它非常清楚。

    假设你希望从纽约旅行到加利福尼亚。

    ![旅行示例](https://www.dailydoseofds.com/content/images/2024/02/image-107.png)

    如果我们遵循NSW方法，这次旅行就像从一个城市到另一个城市旅行，比如通过城际出租车，这需要很多跳跃，但逐渐使我们更接近目的地，如下所示：

    ![城际旅行](https://www.dailydoseofds.com/content/images/2024/02/image-109.png)

    这是最优的吗？

    不对吧？

    现在，想一想。

    你如何更优地覆盖这条路线，或者你在现实生活中如何更优地覆盖这条路线？

    ![飞机动画](https://media.tenor.com/cK657FeMec0AAAAC/riding-plane.gif)

    如果你想到飞行，你的思考方向是正确的。

    将**跳跃列表**视为使用不同交通方式规划旅行的方法，其中一些方式可以在小跳跃中旅行更大的距离。

    所以本质上，我们可以从纽约乘飞机到更接近加利福尼亚的主要城市，比如丹佛，而不是从一个城市跳到另一个城市。

    ![飞行到丹佛](https://www.dailydoseofds.com/content/images/2024/02/image-111.png)

    这次飞行在一次跳跃中覆盖了更长的距离，类似于跳过我们原本会从一个城市到另一个城市覆盖的几个顶点。

    👉 当然，我知道纽约和加利福尼亚之间有直飞航班。这只是为了演示目的，所以假设纽约和加利福尼亚之间没有这样的航班。

    从丹佛，我们可以乘坐另一种更快的交通方式，涉及更少的跳跃，比如火车到达加利福尼亚：

    ![火车到加利福尼亚](https://www.dailydoseofds.com/content/images/2024/02/image-112.png)

    为了增加更多的粒度，假设一旦我们到达加利福尼亚的火车站，我们希望旅行到加利福尼亚洛杉矶的某个地方。

    现在我们需要可以进行较小跳跃的东西，所以出租车在这里是完美的。

    ![出租车动画](https://media.tenor.com/-zBpoKo1-GoAAAAC/taxi-insurance-car-insurance.gif)

    那么我们在这里做了什么？

    这种较长航班、相对较短火车旅行和在城市内旅行的出租车的组合使我们能够在相对很少的停靠点到达目的地。

    这正是跳跃列表所做的。

    跳跃列表是一种数据结构，允许在排序列表中高效搜索元素。它们类似于链表，但具有额外的"跳跃指针"层，允许更快的遍历。

    这是链表的样子：

    ![链表](https://www.dailydoseofds.com/content/images/2024/02/image-115.png)

    在跳跃列表中，每个元素（或节点）包含一个值和一组可以"跳过"列表中几个元素的前向指针。

    ![跳跃列表](https://www.dailydoseofds.com/content/images/2024/02/image-116.png)

    这些前向指针在列表中创建多个层（上面视觉中的`层0`、`层1`、`层2`），每个级别代表不同的"跳跃距离"。

    - 顶层（`层2`）可以被认为是可以在一跳中旅行更长距离的飞行。
    - 中间层（`层1`）可以被认为是可以在一跳中旅行比飞行相对较短距离的火车。
    - 底层（`层0`）可以被认为是可以在一跳中旅行短距离的出租车。

    必须保留在每层中的节点使用概率方法决定。

    基本思想是节点以递减的概率包含在更高层中，导致更高级别的节点更少，而底层总是包含所有节点。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    更具体地说，在跳跃列表构建之前，每个节点被随机分配一个整数`L`，它指示它可以在跳跃列表数据结构中存在的**最大**层。这是这样做的：

    ![层分配公式](https://www.dailydoseofds.com/content/images/2024/02/image-124.png)

    - `uniform(0,1)`生成0和1之间的随机数。
    - `floor()`将结果向下舍入到最近的整数。
    - CLM是调整层之间重叠的层乘数常数。增加此参数会导致更多重叠。

    例如，如果一个节点有`L=2`，这意味着它必须存在于`层2`、`层1`和`层0`上。

    另外，假设层乘数（CLM）设置为`0`。这意味着所有节点的`L=0`：

    ![CLM=0的情况](https://www.dailydoseofds.com/content/images/2024/02/image-125.png)

    如上所述，`L`指示节点可以在跳跃列表数据结构中存在的**最大**层。如果所有节点的`L=0`，这意味着跳跃列表将只有一层。

    增加此参数会导致层之间更多重叠和更多层，如下面的图所示：

    ![CLM分布](https://www.dailydoseofds.com/content/images/2024/02/Clm_distribution.jpeg)

    如上所示：

    - 使用CLM=0，跳跃列表只能有`层0`，这类似于NSW搜索。
    - 使用CLM=0.25，我们得到一个更多的层，大约有6-7个节点。
    - 使用CLM=1，我们得到四层。
    - 在所有情况下，`层0`总是有所有节点。

    目标是为CLM决定一个最优值，因为我们不希望有太多层和如此多的重叠，同时也不希望只有一层（当CLM=0时），这不会导致速度提升改进。

    现在，让我解释跳跃列表如何加速搜索过程。

    假设我们想要在这个列表中找到元素`50`。

    ![搜索元素50](https://www.dailydoseofds.com/content/images/2024/02/image-115.png)

    如果我们使用典型的链表，我们会从第一个元素（`HEAD`）开始，逐个扫描每个节点，看看它是否匹配查询（`50`）。

    看看跳跃列表如何帮助我们优化这个搜索过程。

    我们从顶层（`层2`）开始，检查同一层中下一个节点对应的值，即`65`。

    ![跳跃列表搜索步骤1](https://www.dailydoseofds.com/content/images/2024/02/image-118.png)

    由于`65>50`且它是单向链表，我们必须下降一级。

    在`层1`中，我们检查同一层中下一个节点对应的值，即`36`。

    ![跳跃列表搜索步骤2](https://www.dailydoseofds.com/content/images/2024/02/image-119.png)

    由于`50>36`，明智的做法是移动到对应值`36`的节点。

    现在再次在`层1`中，我们检查同一层中下一个节点对应的值，即`65`。

    ![跳跃列表搜索步骤3](https://www.dailydoseofds.com/content/images/2024/02/image-120.png)

    再次，由于`65>50`且它是单向链表，我们必须下降一级。

    我们到达`层0`，可以按通常的方式遍历。

    如果我们在不构建跳跃列表的情况下遍历链表，我们会花费`5`跳：

    ![普通链表遍历](https://www.dailydoseofds.com/content/images/2024/02/image-121.png)

    但使用跳跃列表，我们在`3`跳中完成了相同的搜索：

    ![跳跃列表遍历](https://www.dailydoseofds.com/content/images/2024/02/image-122.png)

    这很简单和优雅，不是吗？

    虽然将跳跃次数从`5`减少到`3`可能听起来不是很大的改进，但重要的是要注意典型的向量数据库有数百万个节点。

    因此，这种改进很快就会扩展以提供运行时好处。

    ##### → HNSW中的图构建

    现在我们理解了跳跃列表的工作原理，理解分层可导航小世界的图构建过程也相当简单。

    考虑这是我们当前的图结构：

    ![当前图结构](https://www.dailydoseofds.com/content/images/2024/02/image-132.png)

    我理解我们从构建过程的中间开始，但请耐心等待，因为它会澄清一切。

    本质上，上面的图总共有三层，它正在构建中。另外，随着我们向上，节点数量减少，这是跳跃列表中理想情况下发生的。

    现在，假设我们希望插入一个新节点（下图中的蓝色节点），其最大级别（由概率分布确定）是`L=1`。这意味着这个节点将存在于`层1`和`层0`上。

    ![插入新节点](https://www.dailydoseofds.com/content/images/2024/02/image-133.png)

    现在，我们的目标是将这个新节点连接到`层0`和`层1`上图中的其他节点。

    这是我们的做法：

    - 我们从最顶层（`层2`）开始，为这个新节点随机选择一个入口点：

    ![选择入口点](https://www.dailydoseofds.com/content/images/2024/02/image-136.png)

    - 我们探索这个入口点的邻居，并选择最接近要插入的新节点的那个。

    ![探索邻居](https://www.dailydoseofds.com/content/images/2024/02/image-134.png)

    - 为蓝色节点找到的最近邻成为下一层的入口点。因此，我们移动到下一层（`层1`）中最近邻的相应节点：

    ![移动到下一层](https://www.dailydoseofds.com/content/images/2024/02/image-135.png)

    - 这样，我们就到达了必须插入这个蓝色节点的层。

    这里，请注意，如果仍然有更多层，我们会重复上述找到入口点最近邻并向下移动一层的过程，直到我们到达感兴趣的层。

    例如，想象这个节点的最大级别值是`L=0`。因此，蓝色节点只会存在于最底层。

    ![L=0的情况](https://www.dailydoseofds.com/content/images/2024/02/image-137.png)

    移动到`层1`后（如上图所示），我们还没有到达感兴趣的层。

    所以我们探索`层1`中入口点的邻域，再次找到蓝色点的最近邻。

    ![继续探索](https://www.dailydoseofds.com/content/images/2024/02/image-138.png)

    现在，在`层1`上找到的最近邻成为我们`层0`的入口点。

    回到最大级别值为`L=1`的情况。我们目前在`层1`，其中入口点在下图中标记，我们必须在这一层插入蓝色节点：

    ![插入到层1](https://www.dailydoseofds.com/content/images/2024/02/image-140.png)

    要插入节点，我们这样做：

    - 探索当前层中入口点的邻居，并将新节点连接到前`K`个最近邻。为了确定前`K`个邻居，在此步骤中贪婪地探索总共`efConstruction`（超参数）个邻居。例如，如果`K=2`，那么在上图中，我们将蓝色节点连接到以下节点：

    ![连接到K个邻居](https://www.dailydoseofds.com/content/images/2024/02/image-142.png)

    然而，为了确定这些前K=2个节点，我们可能已经探索了，比如`efConstruction=3`个邻居（这样做的目的很快就会变得清楚），如下所示：

    ![探索efConstruction个邻居](https://www.dailydoseofds.com/content/images/2024/02/image-144.png)

    现在，我们必须在低于蓝色节点的最大层值`L`的层插入蓝色节点。

    在这种情况下，我们不像之前那样只保留一个到下一层的入口点，如下所示：

    ![单个入口点](https://www.dailydoseofds.com/content/images/2024/02/image-145.png)

    然而，在上一层中探索的所有`efConstruction`个节点都被视为下一层的入口点。

    一旦我们进入下一层，重复该过程，其中我们通过探索所有`efConstruction`入口节点将蓝色节点连接到前K个邻居。

    当新节点在`级别0`连接到节点时，逐层插入过程结束。

    💡 我故意在这里省略了一些小细节，因为理解起来会太复杂。另外，在任何现实生活情况下，我们几乎不必实现这个算法，因为它已经在使用HNSW进行索引的流行向量数据库中实现。我们必须知道的唯一事情是HNSW在高层次上是如何工作的。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##### → HNSW中的搜索

    考虑在插入所有节点后，我们得到以下图：

    ![完整的HNSW图](https://www.dailydoseofds.com/content/images/2024/02/image-146.png)

    让我们了解近似最近邻搜索是如何工作的。

    假设我们想要在下图中找到黄色向量的最近邻：

    ![搜索目标](https://www.dailydoseofds.com/content/images/2024/02/image-147.png)

    我们从顶层（`层2`）的入口点开始搜索：

    ![开始搜索](https://www.dailydoseofds.com/content/images/2024/02/image-148.png)

    我们探索`A`的连接邻居，看看哪个最接近黄色节点。在这一层中，它是`C`。

    算法贪婪地探索层中顶点的邻域。在此过程中，我们始终朝着查询向量移动。

    当在一层中找不到更接近查询向量的更近节点时，我们移动到下一层，同时将最近邻（在这种情况下是`C`）视为下一层的入口点：

    ![移动到下一层](https://www.dailydoseofds.com/content/images/2024/02/image-149.png)

    邻域探索过程再次重复。

    我们探索`C`的邻居，贪婪地移动到最接近查询向量的特定邻居：

    ![探索C的邻居](https://www.dailydoseofds.com/content/images/2024/02/image-151.png)

    再次，由于`层1`中不存在更接近查询向量的节点，我们移动到下一层，同时将最近邻（在这种情况下是`F`）视为下一层的入口点。

    但这次，我们已经到达了`层0`。因此，在这种情况下将返回近似最近邻。

    当我们移动到`层0`并开始探索其邻域时，我们注意到它没有更接近查询向量的邻居：

    ![最终结果](https://www.dailydoseofds.com/content/images/2024/02/image-152.png)

    因此，对应于节点F的向量作为近似最近邻返回，巧合的是，它也恰好是真正的最近邻。

    ##### → HNSW vs NSW

    在上述搜索过程中，只花费了**`2`跳**（下降不是跳）来返回查询向量的最近邻。

    让我们看看使用NSW找到最近邻需要多少跳。为了简单起见，让我们考虑NSW构建的图是由HNSW图的`层0`表示的：

    ![NSW图](https://www.dailydoseofds.com/content/images/2024/02/image-153.png)

    我们之前从节点`A`作为入口点开始，所以让我们在这里也考虑相同的。

    我们从节点`A`开始，探索其邻居，并移动到节点`E`，因为那是最接近查询向量的：

    ![NSW搜索步骤1](https://www.dailydoseofds.com/content/images/2024/02/image-154.png)

    从节点`E`，我们移动到节点`B`，它比节点`E`更接近查询向量。

    ![NSW搜索步骤2](https://www.dailydoseofds.com/content/images/2024/02/image-155.png)

    接下来，我们探索节点`B`的邻居，注意到节点`I`最接近查询向量，所以我们现在跳到那个节点：

    ![NSW搜索步骤3](https://www.dailydoseofds.com/content/images/2024/02/image-157.png)

    由于节点`I`找不到任何其他它连接的更接近自己的节点，算法返回节点`I`作为最近邻。

    那里发生了什么？

    算法不仅花费了更多跳（`3`）来返回最近邻，而且还返回了不太优的最近邻。

    另一方面，HNSW花费了更少的跳并返回了更准确和最优的最近邻。

    完美！

    这样，你已经学会了五种相当常见的索引策略来索引向量数据库中的向量以进行高效搜索。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 在大语言模型（LLM）中使用向量数据库

    此时，一个有趣的学习内容是大语言模型（LLM）如何确切地利用向量数据库。

    根据我的经验，许多人面临的最大困惑是以下问题：

    > 一旦我们训练了我们的LLM，它将有一些用于文本生成的模型权重。向量数据库在这里如何适配？

    ![LLM困惑](https://www.dailydoseofds.com/content/images/2024/02/image-242.png)

    在我看来，这是一个相当真实的查询。

    让我解释向量数据库如何帮助LLM在它们产生的内容方面更加准确和可靠。

    首先，我们必须理解LLM是在学习了训练期间提供的语料库的静态版本后部署的。

    ![静态训练数据](https://www.dailydoseofds.com/content/images/2024/02/image-243.png)

    例如，如果模型在考虑到`2024年1月31日`之前的数据后部署，我们在训练后一周使用它，它将不知道那些天发生了什么。

    ![知识截止](https://www.dailydoseofds.com/content/images/2024/02/image-244.png)

    每天在新数据上重复训练新模型（或适应最新版本）是不切实际和成本无效的。事实上，LLM可能需要数周才能训练。

    另外，如果我们开源了LLM，其他人想要在他们的私有数据集上使用它，这当然在训练期间没有显示过，会怎么样？

    如预期的那样，LLM将对此一无所知。

    ![私有数据问题](https://www.dailydoseofds.com/content/images/2024/02/image-245.png)

    但如果你想想，训练LLM了解世界上的每一件事真的是我们的目标吗？

    **一个大大的不！**

    那不是我们的目标。

    相反，它更多的是帮助LLM学习语言的整体结构，以及如何理解和生成它。

    ![语言理解](https://www.dailydoseofds.com/content/images/2024/02/image-246.png)

    所以，一旦我们在一个荒谬的大训练语料库上训练了这个模型，可以预期模型将具有相当水平的语言理解和生成能力。

    因此，如果我们能够找到一种方法让LLM查找它们没有训练过的新信息并在文本生成中使用它（**无需再次训练模型**），那就太好了！

    一种方法可能是在提示本身中提供该信息。

    换句话说，如果不希望训练或微调模型，我们可以在给LLM的提示中提供所有必要的细节。

    ![提示中的信息](https://www.dailydoseofds.com/content/images/2024/02/image-248.png)

    不幸的是，这只适用于少量信息。

    这是因为LLM是自回归模型。

    💡 **自回归模型**是那些一次生成一个步骤输出的模型，其中每个步骤都依赖于前面的步骤。在LLM的情况下，这意味着模型一次生成一个单词的文本，**基于它已经生成的单词**。

    因此，由于LLM考虑先前的单词，它们有一个在提示中实际上不能超过的令牌限制。

    总的来说，这种在提示中提供一切的方法并不那么有前途，因为它将实用性限制在几千个令牌，而在现实生活中，附加信息可能有数百万个令牌。

    **这就是向量数据库帮助的地方。**

    我们可以利用向量数据库动态更新模型对世界的理解，而不是每次出现新数据或变化时都重新训练LLM。

    如何？

    很简单。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    如本文前面所讨论的，向量数据库帮助我们以向量的形式存储信息，其中每个向量捕获被编码文本片段的语义信息。

    因此，我们可以通过使用嵌入模型将信息编码为向量来在向量数据库中维护我们的可用信息。

    ![信息编码](https://www.dailydoseofds.com/content/images/2024/02/image-249.png)

    当LLM需要访问这些信息时，它可以使用提示向量的相似性搜索查询向量数据库。

    更具体地说，相似性搜索将尝试在向量数据库中找到与输入查询向量相似的内容。

    ![相似性搜索](https://www.dailydoseofds.com/content/images/2024/02/image-250.png)

    这就是索引变得重要的地方，因为我们的向量数据库可能有数百万个向量。

    理论上，我们可以将输入向量与向量数据库中的每个向量进行比较。

    但为了实际实用性，我们必须尽快找到最近邻。

    这就是为什么我们之前讨论的索引技术变得如此重要。它们帮助我们几乎实时地找到近似最近邻。

    继续，一旦检索到近似最近邻，我们收集生成这些特定向量的上下文。这是可能的，因为向量数据库不仅存储向量，还存储生成这些向量的原始数据。

    ![检索上下文](https://www.dailydoseofds.com/content/images/2024/02/image-253.png)

    这个搜索过程检索与查询向量相似的上下文，查询向量代表LLM感兴趣的上下文或主题。

    我们可以将这个检索到的内容与用户提供的实际提示一起增强，并将其作为输入给LLM。

    ![增强提示](https://www.dailydoseofds.com/content/images/2024/02/image-254.png)

    因此，LLM可以在生成文本时轻松地整合这些信息，因为它现在在提示中有相关的细节可用。

    恭喜！

    你刚刚学会了**检索增强生成（RAG）**。我相信你现在一定听过这个术语很多次，我们上面讨论的就是RAG背后的整个想法。

    我故意没有在之前的任何地方提到RAG，以建立所需的流程并避免首先用这个术语吓到你。

    事实上，甚至它的名字也完全证明了我们用这种技术做什么：

    - **检索**：从知识源（如数据库或内存）访问和检索信息。
    - **增强**：用额外信息或上下文增强或丰富某些东西，在这种情况下是文本生成过程。
    - **生成**：创建或产生某些东西的过程，在这种情况下，生成文本或语言。

    RAG的另一个关键优势是它大大帮助LLM减少其响应中的**幻觉**。我相信你一定在某个地方也听过这个术语。

    当语言模型生成不基于现实的信息或编造事情时，就会发生幻觉。

    这可能导致模型生成不正确或误导性的信息，这在许多应用中可能是有问题的。

    使用RAG，语言模型可以使用从向量数据库检索的信息（预期是可靠的）来确保其响应基于现实世界的知识和上下文，减少幻觉的可能性。

    这使得模型的响应更加准确、可靠和上下文相关，提高了其整体性能和实用性。

    这个想法也很直观。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 向量数据库提供商

    如今，有大量的向量数据库提供商，可以帮助我们高效地存储和检索数据的向量表示。

    - **[Pinecone](https://www.pinecone.io/)**：Pinecone是一个托管的向量数据库服务，提供快速、可扩展和高效的向量数据存储和检索。它为构建AI应用程序提供了一系列功能，如相似性搜索和实时分析。

    - **[Weaviate](https://github.com/weaviate/weaviate)**：Weaviate是一个**开源向量数据库**，它是强大的、可扩展的、云原生的和快速的。使用Weaviate，可以使用最先进的ML模型将文本、图像等转换为可搜索的向量数据库。

    - **[Milvus](https://github.com/milvus-io/milvus)**：Milvus是一个开源向量数据库，旨在为嵌入相似性搜索和AI应用程序提供动力。Milvus使非结构化数据搜索更加可访问，并提供一致的用户体验，无论部署环境如何。

    - **[Qdrant](https://github.com/qdrant/qdrant)**：Qdrant是一个向量相似性搜索引擎和向量数据库。它提供了一个生产就绪的服务，具有方便的API来存储、搜索和管理点——带有额外有效载荷的向量。Qdrant专为扩展过滤支持而定制。它使其对所有类型的神经网络或基于语义的匹配、分面搜索和其他应用程序都很有用。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Pinecone 演示

    接下来，让我们深入了解如何在向量数据库中索引向量并对其执行搜索操作的一些实际细节。

    对于这个演示，我将使用Pinecone，因为它可能是最容易开始和理解的之一。但上面链接了许多其他提供商，如果你愿意，可以探索。

    首先，我们安装一些依赖项，如Pinecone和Sentence transformers：
    """
    )
    return


@app.cell
def _():
    # 安装依赖项
    # !pip install pinecone-client sentence-transformers
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    编码这个文本数据集为向量嵌入以将它们存储在向量数据库中是很重要的。为此，我们将利用`SentenceTransformers`库。

    它提供预训练的基于transformer的架构，可以有效地将文本编码为密集向量表示，通常称为嵌入。

    SentenceTransformers模型提供各种预训练架构，如`BERT`、`RoBERTa`和`DistilBERT`，专门为句子嵌入进行微调。

    这些嵌入捕获文本输入之间的语义相似性和关系，使它们适用于分类和聚类等下游任务。

    DistilBERT是一个相对较小的模型，所以我们将在这个演示中使用它。

    接下来，打开一个Jupyter Notebook并导入上述库：
    """
    )
    return


@app.cell
def _():
    # 导入必要的库
    import numpy as np
    import pinecone
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer, np


@app.cell
def _(mo):
    mo.md(r"""继续，我们下载并实例化DistilBERT句子transformer模型如下：""")
    return


@app.cell
def _(SentenceTransformer):
    # 下载并实例化DistilBERT模型
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    return (model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    要开始使用Pinecone并创建向量数据库，我们需要一个Pinecone API密钥。

    要获得这个，请前往Pinecone网站并在这里创建一个账户：[https://app.pinecone.io/?sessionType=signup](https://app.pinecone.io/?sessionType=signup)。注册后我们到达以下页面：

    ![Pinecone仪表板](https://www.dailydoseofds.com/content/images/2024/02/Screenshot-2024-02-16-at-11.20.12-PM.png)

    从下面仪表板的左侧面板获取你的API密钥：

    ![API密钥位置](https://www.dailydoseofds.com/content/images/2024/02/Screenshot-2024-02-16-at-11.37.20-PM.png)

    点击`API Keys` -> `Create API Key` -> `Enter API Key Name` -> `Create`。

    ![创建API密钥](https://www.dailydoseofds.com/content/images/2024/02/Screenshot-2024-02-16-at-11.40.12-PM.png)

    完成！

    获取这个API密钥（如复制到剪贴板按钮），回到Jupyter Notebook，并使用这个API密钥建立到Pinecone的连接，如下所示：
    """
    )
    return


@app.cell
def _():
    # 建立到Pinecone的连接
    # 注意：在实际使用中，请将API密钥存储在环境变量中
    # pinecone.init(api_key="your-api-key-here", environment="your-environment")

    # 为了演示目的，我们将跳过实际的Pinecone连接
    print("Pinecone连接已建立（演示模式）")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    在Pinecone中，我们将向量嵌入存储在[索引](https://docs.pinecone.io/docs/indexes)中。我们创建的任何索引中的向量必须共享相同的维度和用于测量相似性的距离度量。

    我们使用上面创建的`Pinecone`类对象的`create_index()`方法创建索引。

    顺便说一下，目前，由于我们没有索引，运行`list_indexes()`方法在字典的`indexes`键中返回一个空列表：

    ![空索引列表](https://www.dailydoseofds.com/content/images/2024/02/image-158.png)

    回到创建索引，我们使用`Pinecone`类对象的`create_index()`方法如下：
    """
    )
    return


@app.cell
def _():
    # 创建索引（演示代码）
    # pc.create_index(
    #     name="vector-db-demo",
    #     dimension=768,  # DistilBERT的嵌入维度
    #     metric="euclidean",
    #     spec=PodSpec(environment="gcp-starter")
    # )

    print("索引创建代码（演示模式）")
    print("- 名称: vector-db-demo")
    print("- 维度: 768")
    print("- 度量: euclidean")
    print("- 环境: gcp-starter")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    这个函数调用的分解：

    - `name`：索引的名称。这是一个用户定义的名称，可以在以后对索引执行操作时用来引用索引。
    - `dimension`：将存储在索引中的向量的维度。这应该与将插入索引的向量的维度匹配。我们在这里指定了`768`，因为这是`SentenceTransformer`模型返回的嵌入维度。
    - `metric`：用于计算向量之间相似性的距离度量。在这种情况下，使用`euclidean`，这意味着欧几里得距离将用作相似性度量。
    - `spec`：指定将创建索引的环境的`PodSpec`对象。在这个例子中，索引在名为`gcp-starter`的GCP（Google Cloud Platform）环境中创建。

    执行此方法创建一个索引，我们也可以在仪表板中看到：

    ![索引创建成功](https://www.dailydoseofds.com/content/images/2024/02/Screenshot-2024-02-17-at-12.01.56-AM.png)

    现在我们已经创建了一个索引，我们可以推送向量嵌入。

    为了做到这一点，让我们创建一些文本数据并使用`SentenceTransformer`模型对其进行编码。

    我在下面创建了一些虚拟数据：
    """
    )
    return


@app.cell
def _():
    # 创建示例文本数据
    data = [
        {"id": "1", "text": "向量数据库对于AI应用程序非常有用"},
        {"id": "2", "text": "机器学习模型需要大量的训练数据"},
        {"id": "3", "text": "深度学习是人工智能的一个子领域"},
        {"id": "4", "text": "自然语言处理帮助计算机理解人类语言"},
        {"id": "5", "text": "计算机视觉使机器能够解释视觉信息"},
        {"id": "6", "text": "推荐系统使用协同过滤技术"},
        {"id": "7", "text": "大数据分析需要强大的计算资源"},
        {"id": "8", "text": "云计算提供可扩展的基础设施"},
        {"id": "9", "text": "区块链技术确保数据的安全性"},
        {"id": "10", "text": "物联网连接各种智能设备"}
    ]

    print(f"创建了 {len(data)} 个文本样本")
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""我们为这些句子创建嵌入如下：""")
    return


@app.cell
def _(data, model):
    # 为文本数据创建嵌入
    vector_data = []

    for item in data:
        # 为文本生成嵌入
        embedding = model.encode(item["text"])

        # 创建向量信息字典
        vector_info1 = {
            "id": item["id"],
            "values": embedding.tolist()  # 转换为列表格式
        }

        vector_data.append(vector_info1)

    print(f"为 {len(vector_data)} 个文本生成了嵌入")
    print(f"每个嵌入的维度: {len(vector_data[0]['values'])}")
    return (vector_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    这个代码片段遍历我们之前定义的`data`列表中的每个句子，并使用下载的句子transformer模型（`model`）将每个句子的文本编码为向量。

    然后它创建一个包含句子ID（`id`）和相应向量（`values`）的字典`vector_info`，并将此字典附加到`vector_data`列表中。

    在实际实例中，同一账户下可能有多个索引，我们必须创建一个`index`对象，指定我们希望将这些嵌入添加到的索引。这样做如下：
    """
    )
    return


@app.cell
def _():
    # 创建索引对象（演示代码）
    # index = pc.Index("vector-db-demo")

    print("索引对象创建（演示模式）")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    现在我们有了嵌入和索引，我们**upsert**这些向量。

    Upsert是一个数据库操作，结合了**update**和**insert**的动作。如果文档不存在，它将新文档插入集合中，如果文档存在，则更新现有文档。Upsert是数据库中的常见操作，特别是在NoSQL数据库中，它用于确保文档根据其在集合中的存在而被插入或更新。
    """
    )
    return


@app.cell
def _(vector_data):
    # Upsert向量到索引（演示代码）
    # index.upsert(vectors=vector_data)

    print("向量upsert操作（演示模式）")
    print(f"准备upsert {len(vector_data)} 个向量")

    # 显示第一个向量的信息作为示例
    if vector_data:
        first_vector = vector_data[0]
        print(f"示例向量 - ID: {first_vector['id']}, 维度: {len(first_vector['values'])}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    完成！

    我们已经将这些向量添加到索引中。虽然输出确实突出了这一点，我们可以通过使用[`describe_index_stats`](https://docs.pinecone.io/reference/describe_index_stats)操作来双重验证这一点，检查当前向量计数是否与我们upsert的向量数量匹配：
    """
    )
    return


@app.cell
def _(vector_data):
    # 描述索引统计（演示代码）
    # stats = index.describe_index_stats()

    # 模拟统计信息
    stats = {
        'dimension': 768,
        'index_fullness': 0.0,
        'namespaces': {'': {'vector_count': len(vector_data)}},
        'total_vector_count': len(vector_data)
    }

    print("索引统计信息（演示模式）:")
    print(f"- 维度: {stats['dimension']}")
    print(f"- 索引满度: {stats['index_fullness']}")
    print(f"- 总向量数: {stats['total_vector_count']}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    返回字典中每个键的含义：

    - `dimension`：索引中存储的向量的维度（在这种情况下为`768`）。
    - `index_fullness`：索引满度的度量，通常表示索引中被占用的槽位百分比。
    - `namespaces`：包含索引中每个命名空间统计信息的字典。在这种情况下，只有一个命名空间（''），`vector_count`为`10`，表示索引中有`10`个向量。
    - `total_vector_count`：索引中所有命名空间的向量总数（在这种情况下为`10`）。

    现在我们已经在上述索引中存储了向量，让我们运行相似性搜索以查看获得的结果。

    我们可以使用我们之前创建的`index`对象的`query()`方法来做到这一点。

    首先，我们定义一个搜索文本并生成其嵌入：
    """
    )
    return


@app.cell
def _(model):
    # 定义搜索文本并生成嵌入
    search_text = "向量数据库真的很有帮助"
    search_embedding = model.encode(search_text)

    print(f"搜索文本: {search_text}")
    print(f"搜索嵌入维度: {len(search_embedding)}")
    return (search_embedding,)


@app.cell
def _(mo):
    mo.md(r"""接下来，我们这样查询：""")
    return


@app.cell
def _(data, np, search_embedding, vector_data):
    # 查询索引（演示代码）
    # query_result = index.query(vector=search_embedding.tolist(), top_k=3, include_metadata=True)

    # 模拟查询结果 - 计算与搜索嵌入的相似性

    similarities = []
    for i, vector_info in enumerate(vector_data):
        # 计算余弦相似性
        vec1 = np.array(search_embedding)
        vec2 = np.array(vector_info['values'])

        # 余弦相似性
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        similarities.append({
            'id': vector_info['id'],
            'score': float(cosine_sim),
            'text': data[i]['text']
        })

    # 按相似性分数排序并取前3个
    similarities.sort(key=lambda x: x['score'], reverse=True)
    top_3 = similarities[:3]

    # 模拟Pinecone查询结果格式
    query_result = {
        'matches': [
            {'id': item['id'], 'score': item['score']} for item in top_3
        ],
        'namespace': '',
        'usage': {'read_units': 5}
    }

    print("查询结果（演示模式）:")
    print(f"找到 {len(query_result['matches'])} 个匹配项")

    for i, match in enumerate(top_3):
        print(f"{i+1}. ID: {match['id']}, 相似性分数: {match['score']:.4f}")
        print(f"   文本: {match['text']}")
        print()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    这个代码片段在索引对象上调用`query`方法，它为给定的查询向量（`search_embedding`）执行最近邻搜索并返回前`3`个匹配项。

    返回字典中每个键的含义：

    - `matches`：字典列表，其中每个字典包含匹配向量的信息。每个字典包括匹配向量的`id`，以及表示查询向量和匹配向量之间相似性的`score`。由于我们在创建此索引时指定了`euclidean`作为我们的度量，较高的分数表示更多的距离，这反过来意味着较少的相似性。
    - `namespace`：执行查询的索引的命名空间。在这种情况下，命名空间是空字符串（''），表示默认命名空间。
    - `usage`：包含查询操作期间资源使用信息的字典。在这种情况下，`read_units`表示查询操作消耗的读取单元数，为`5`。然而，我们最初向此索引附加了`10`个向量，这表明它没有查看所有向量来找到最近邻。

    从上述结果中，我们注意到`search_text`（`"向量数据库真的很有帮助"`）的前3个邻居是相关的文本片段。

    太棒了！
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 结论

    这样，我们就结束了对向量数据库的深度解析。

    总结一下，我们了解到向量数据库是专门设计用于高效存储和检索数据向量表示的专用数据库。

    通过将向量组织到索引中，向量数据库能够进行快速准确的相似性搜索，使它们对于推荐系统和信息检索等任务非常有价值。

    此外，Pinecone演示展示了使用Pinecone服务创建和查询向量索引是多么容易。

    在我结束这篇文章之前，有一个重要的点我想提到。

    仅仅因为向量数据库听起来很酷，并不意味着你必须在每个希望找到向量相似性的地方都采用它们。

    评估是否有必要为你的特定用例使用向量数据库是非常重要的。

    对于具有有限向量数量的小规模应用程序，像NumPy数组和进行穷尽搜索这样的简单解决方案就足够了。

    ![简单解决方案](https://www.dailydoseofds.com/content/images/2024/02/image-255.png)

    除非你看到任何好处，如应用程序中的延迟改进、成本降低等，否则没有必要转向向量数据库。

    我希望你今天学到了新东西！

    我知道我们在这次深度解析中讨论了许多细节，所以如果有任何困惑，请随时在评论中发布。

    或者，如果你希望私下联系，请随时在这里发起聊天：

    ![联系方式](https://www.dailydoseofds.com/content/images/2023/07/Screenshot-2023-07-28-at-3.36.00-PM.png)

    感谢阅读！
    """
    )
    return


if __name__ == "__main__":
    app.run()

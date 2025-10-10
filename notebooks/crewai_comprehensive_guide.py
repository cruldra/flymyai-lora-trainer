import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CrewAI 综合指南

    CrewAI 是一个快速、灵活的多智能体自动化框架,专为编排角色扮演的自主AI智能体而设计。

    本指南将带你深入了解 CrewAI 的核心概念、功能和实际应用。
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    from dotenv import load_dotenv
    import nest_asyncio

    # 允许嵌套事件循环
    nest_asyncio.apply()

    # 加载 .env 文件
    load_dotenv()

    # 检查环境变量
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY 已加载")
    else:
        print("⚠️ OPENAI_API_KEY 未设置")

    if os.getenv("OPENAI_API_BASE_URL"):
        print(f"✅ OPENAI_API_BASE_URL: {os.getenv('OPENAI_API_BASE_URL')}")

    return load_dotenv, mo, nest_asyncio, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. 安装与配置

    CrewAI 是一个独立的框架,不依赖于 LangChain 或其他智能体框架。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 安装命令

    ```bash
    # 基础安装
    pip install crewai

    # 包含额外工具的完整安装
    pip install 'crewai[tools]'
    ```

    ### 环境要求

    - Python >= 3.10, < 3.14
    - 推荐使用 UV 进行依赖管理
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    installation_check = mo.md(
        """
        ### 检查安装状态

        运行下面的代码检查 CrewAI 是否已安装:
        """
    )
    installation_check
    return


@app.cell
def _():
    try:
        import crewai
        print(f"✅ CrewAI 已安装,版本: {crewai.__version__}")
    except ImportError:
        print("❌ CrewAI 未安装,请运行: pip install crewai")
    return (crewai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. 核心概念

    CrewAI 提供两种强大且互补的方法来构建复杂的AI应用:

    ### Crews (团队)
    - 具有真正自主性和代理能力的AI智能体团队
    - 通过基于角色的协作共同完成复杂任务
    - 智能体之间自然、自主的决策
    - 动态任务委派和协作

    ### Flows (流程)
    - 生产就绪的事件驱动工作流
    - 对复杂自动化提供精确控制
    - 安全、一致的状态管理
    - 条件分支支持复杂业务逻辑
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Agents (智能体)

    智能体是 CrewAI 的核心组件,每个智能体都有特定的角色、目标和背景故事。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 智能体的关键属性

    - **role**: 智能体在团队中的角色
    - **goal**: 智能体要实现的目标
    - **backstory**: 智能体的背景故事,影响其行为方式
    - **llm**: 使用的语言模型
    - **tools**: 智能体可以使用的工具列表
    - **verbose**: 是否输出详细日志
    - **allow_delegation**: 是否允许委派任务给其他智能体
    - **memory**: 是否启用记忆功能
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    agent_example_intro = mo.md(
        """
        ### 创建智能体示例

        下面演示如何创建一个研究员智能体:
        """
    )
    agent_example_intro
    return


@app.cell
def _(crewai, os):
    # 配置 LLM
    llm = crewai.LLM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL")
    )

    # 创建一个研究员智能体
    researcher = crewai.Agent(
        role="高级研究专家",
        goal="查找关于AI技术的最新信息和深入见解",
        backstory="""你是一位经验丰富的研究专家,擅长从各种来源查找相关信息。
        你能够以清晰、结构化的方式组织信息,使复杂的主题变得易于理解。""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    print("✅ 已创建研究员智能体")
    print(f"- 角色: {researcher.role}")
    print(f"- 目标: {researcher.goal}")
    return llm, researcher


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Tasks (任务)

    任务定义了智能体需要完成的具体工作。每个任务都有明确的描述和预期输出。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 任务的关键属性

    - **description**: 任务的详细描述
    - **expected_output**: 预期的输出格式和内容
    - **agent**: 执行此任务的智能体
    - **context**: 任务依赖的其他任务(用于任务链)
    - **output_file**: 输出文件路径(可选)
    - **async_execution**: 是否异步执行
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    task_example_intro = mo.md(
        """
        ### 创建任务示例

        下面演示如何创建一个研究任务:
        """
    )
    task_example_intro
    return


@app.cell
def _(crewai, researcher):
    # 创建一个研究任务
    research_task = crewai.Task(
        description="""对人工智能在医疗保健领域的应用进行全面研究。
        重点关注:
        1. 关键概念和定义
        2. 最新发展和趋势
        3. 主要挑战和机遇
        4. 实际应用案例

        确保以结构化的格式组织你的发现。""",
        expected_output="""一份全面的研究文档,包含关于AI在医疗保健领域应用的所有关键信息。
        包括具体的事实、数据和示例。""",
        agent=researcher
    )

    print("✅ 已创建研究任务")
    print(f"- 描述: {research_task.description[:100]}...")
    print(f"- 执行者: {research_task.agent.role}")
    return (research_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Crews (团队)

    Crew 是智能体的集合,它们协同工作以完成一系列任务。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Crew 的关键属性

    - **agents**: 团队中的智能体列表
    - **tasks**: 要执行的任务列表
    - **process**: 执行流程(sequential 或 hierarchical)
    - **verbose**: 是否输出详细日志
    - **memory**: 是否启用团队记忆
    - **manager_llm**: 层级流程中管理者使用的LLM
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 执行流程类型

    #### Sequential Process (顺序流程)
    - 任务按顺序执行
    - 每个任务完成后才开始下一个
    - 适合有明确依赖关系的任务链

    #### Hierarchical Process (层级流程)
    - 自动分配一个管理者智能体
    - 管理者负责协调和委派任务
    - 适合复杂的多智能体协作场景
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    crew_example_intro = mo.md(
        """
        ### 创建 Crew 示例

        下面演示如何创建一个完整的研究团队:
        """
    )
    crew_example_intro
    return


@app.cell
def _(crewai, llm, mo, research_task, researcher):
    # 创建分析师智能体
    analyst = crewai.Agent(
        role="数据分析师和报告撰写者",
        goal="分析研究结果并创建结构良好的综合报告",
        backstory="""你是一位技能娴熟的分析师,擅长数据解释和技术写作。
        你能够从研究数据中识别模式并提取有意义的见解。""",
        llm=llm,
        verbose=True
    )

    # 创建分析任务
    analysis_task = crewai.Task(
        description="""审查研究结果并创建一份综合报告。
        报告应包括:
        1. 执行摘要
        2. 研究中的所有关键信息
        3. 趋势和模式的深入分析
        4. 建议或未来考虑事项""",
        expected_output="""一份精心制作的专业报告,呈现研究结果并附加分析和见解。""",
        agent=analyst,
        context=[research_task]
    )

    # 创建 Crew
    research_crew = crewai.Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=crewai.Process.sequential,
        verbose=True
    )

    print("✅ 已创建研究团队")
    print(f"- 智能体数量: {len(research_crew.agents)}")
    print(f"- 任务数量: {len(research_crew.tasks)}")
    print("- 执行流程: Sequential")
    return analysis_task, analyst, research_crew


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 运行 Crew 示例

    下面的代码演示如何运行团队(需要设置 OPENAI_API_KEY 环境变量):
    """
    )
    return


@app.cell
def _(research_crew):
    result = research_crew.kickoff()
    print(result.raw)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Flows (流程)

    Flows 提供了事件驱动的工作流控制,可以精确管理复杂的自动化流程。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Flow 的核心特性

    - **@start()**: 定义流程的起始点
    - **@listen()**: 监听特定事件或方法的完成
    - **@router()**: 根据条件路由到不同的执行路径
    - **or_()** 和 **and_()**: 组合多个条件
    - **状态管理**: 使用 Pydantic 模型管理流程状态
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    flow_example_intro = mo.md(
        """
        ### Flow 示例代码

        下面展示一个市场分析流程的示例:
        """
    )
    flow_example_intro
    return


@app.cell
def _():
    from crewai.flow.flow import Flow, listen, start, router
    from pydantic import BaseModel

    class MarketState(BaseModel):
        sentiment: str = "neutral"
        confidence: float = 0.0
        recommendations: list = []

    class AnalysisFlow(Flow[MarketState]):
        @start()
        def fetch_data(self):
            self.state.sentiment = "analyzing"
            print(f"📊 开始数据获取, 状态: {self.state.sentiment}")
            return {"sector": "tech", "timeframe": "1W"}

        @listen(fetch_data)
        def analyze(self, data):
            print(f"🔍 分析数据: {data}")
            self.state.confidence = 0.85
            print(f"✅ 分析完成, 置信度: {self.state.confidence}")
            return "analysis_complete"

        @router(analyze)
        def determine_next_steps(self):
            if self.state.confidence > 0.8:
                print("🎯 高置信度路径")
                return "high_confidence"
            print("⚠️ 低置信度路径")
            return "low_confidence"

        @listen("high_confidence")
        def execute_strategy(self):
            self.state.recommendations.append("Execute trade")
            print(f"💼 执行策略: {self.state.recommendations}")
            return "strategy_executed"

    print("✅ 已定义 AnalysisFlow 类")
    return AnalysisFlow, BaseModel, Flow, MarketState, listen, router, start


@app.cell
def _(AnalysisFlow, MarketState):
    # 创建并运行 Flow
    flow = AnalysisFlow()
    flow.kickoff()

    print("\n📋 最终状态:")
    print(f"- 情绪: {flow.state.sentiment}")
    print(f"- 置信度: {flow.state.confidence}")
    print(f"- 建议: {flow.state.recommendations}")
    return (flow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. 工具集成

    CrewAI 支持丰富的工具生态系统,让智能体能够执行各种任务。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 常用工具类别

    #### 文件和文档
    - FileReadTool: 读取文件内容
    - DirectoryReadTool: 读取目录结构
    - PDFSearchTool: 搜索PDF文档

    #### 网络搜索和浏览
    - SerperDevTool: Google搜索
    - ScrapeWebsiteTool: 网页抓取
    - WebsiteSearchTool: 网站搜索

    #### 数据库和数据
    - CSVSearchTool: CSV文件搜索
    - JSONSearchTool: JSON数据搜索
    - DatabaseTool: 数据库查询

    #### AI和机器学习
    - CodeInterpreterTool: 代码执行
    - VisionTool: 图像分析
    - DALLETool: 图像生成
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    tool_example_intro = mo.md(
        """
        ### 使用工具示例

        下面演示如何为智能体配置工具:
        """
    )
    tool_example_intro
    return


@app.cell
def _(crewai, llm, mo):
    try:
        from crewai_tools import SerperDevTool, FileReadTool

        # 创建工具实例
        search_tool = SerperDevTool()
        file_tool = FileReadTool()

        # 创建带工具的智能体
        researcher_with_tools = crewai.Agent(
            role="网络研究员",
            goal="使用搜索工具查找最新信息",
            backstory="你是一位擅长使用各种工具进行研究的专家。",
            llm=llm,
            tools=[search_tool, file_tool],
            verbose=True
        )

        print("✅ 已创建带工具的智能体")
        print(f"- 工具数量: {len(researcher_with_tools.tools)}")
        print(f"- 工具列表: {[type(t).__name__ for t in researcher_with_tools.tools]}")
    except ImportError:
        print("⚠️ 需要安装 crewai-tools: pip install 'crewai[tools]'")
    return FileReadTool, SerperDevTool, file_tool, researcher_with_tools, search_tool


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 创建自定义工具

    使用 `@tool` 装饰器可以轻松创建自定义工具:
    """
    )
    return


@app.cell
def _(crewai, llm):
    from crewai.tools import tool

    @tool("计算器工具")
    def calculator(operation: str) -> str:
        """执行基本的数学运算。

        Args:
            operation: 要执行的数学表达式,如 "2 + 2"

        Returns:
            计算结果的字符串表示
        """
        try:
            result = eval(operation)
            return f"结果是: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 使用自定义工具
    agent_with_calculator = crewai.Agent(
        role="数学助手",
        goal="帮助用户进行数学计算",
        backstory="你是一位精通数学的助手,能够快速准确地进行各种数学计算。",
        llm=llm,
        tools=[calculator],
        verbose=True
    )

    # 创建一个使用计算器的任务
    calc_task = crewai.Task(
        description="计算以下数学表达式的结果: 10 + 5 * 2",
        expected_output="数学表达式的计算结果",
        agent=agent_with_calculator
    )

    # 创建 Crew 并执行
    calc_crew = crewai.Crew(
        agents=[agent_with_calculator],
        tasks=[calc_task],
        verbose=True
    )

    print("✅ 已创建自定义计算器工具")
    print(f"- 工具名称: {calculator.name}")
    print(f"- 智能体角色: {agent_with_calculator.role}")
    print("\n🔧 执行计算任务...")

    calc_result = calc_crew.kickoff()
    print(f"\n📊 计算结果: {calc_result.raw}")

    return agent_with_calculator, calc_crew, calc_result, calc_task, calculator, tool


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. 实战示例

    下面是一些实用的 CrewAI 应用场景。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 示例 1: 内容创作团队

    创建一个包含作家、编辑和SEO专家的内容创作团队:
    """
    )
    return


@app.cell
def _(crewai, llm, mo):
    # 内容创作团队
    writer = crewai.Agent(
        role="内容作家",
        goal="创作引人入胜的高质量内容",
        backstory="你是一位经验丰富的内容作家,擅长创作吸引读者的文章。",
        llm=llm,
        verbose=True
    )

    editor = crewai.Agent(
        role="编辑",
        goal="审查和改进内容质量",
        backstory="你是一位细心的编辑,能够发现并修正内容中的问题。",
        llm=llm,
        verbose=True
    )

    seo_expert = crewai.Agent(
        role="SEO专家",
        goal="优化内容以提高搜索引擎排名",
        backstory="你是SEO领域的专家,了解如何让内容在搜索引擎中脱颖而出。",
        llm=llm,
        verbose=True
    )

    print("✅ 已创建内容创作团队")
    print("团队成员:")
    print("- 内容作家")
    print("- 编辑")
    print("- SEO专家")
    return editor, seo_expert, writer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 示例 2: 客户服务自动化

    使用层级流程创建多层客户服务系统:
    """
    )
    return


@app.cell
def _(crewai, llm):
    # 客户服务团队
    triage_agent = crewai.Agent(
        role="客服分流专员",
        goal="快速识别客户问题类型并分配给合适的专员",
        backstory="你擅长快速理解客户需求并做出正确的分流决策。",
        llm=llm,
        verbose=True
    )

    technical_support = crewai.Agent(
        role="技术支持专员",
        goal="解决技术相关的客户问题",
        backstory="你是技术支持专家,能够解决各种技术问题。",
        llm=llm,
        verbose=True
    )

    billing_support = crewai.Agent(
        role="账单支持专员",
        goal="处理账单和付款相关的问题",
        backstory="你精通账单系统,能够快速解决付款问题。",
        llm=llm,
        verbose=True
    )

    print("✅ 已创建客户服务团队")
    print("团队成员:")
    print("- 客服分流专员")
    print("- 技术支持专员")
    print("- 账单支持专员")
    return billing_support, technical_support, triage_agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9. 高级特性

    ### 记忆系统
    CrewAI 支持多种记忆类型:
    - **短期记忆**: 当前对话上下文
    - **长期记忆**: 跨会话的持久化记忆
    - **实体记忆**: 关于特定实体的记忆

    ### 协作模式
    - **任务委派**: 智能体可以将任务委派给其他智能体
    - **信息共享**: 智能体之间可以共享信息和上下文
    - **反馈循环**: 支持迭代改进的反馈机制

    ### 人机协作 (HITL)
    - 在关键决策点请求人类输入
    - 人类可以审查和修改智能体的输出
    - 支持异步的人类反馈流程
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 10. 最佳实践

    ### 智能体设计
    1. **明确角色定义**: 给每个智能体清晰的角色和职责
    2. **具体的目标**: 设定可衡量、可实现的目标
    3. **丰富的背景故事**: 背景故事影响智能体的行为方式
    4. **合适的工具**: 为智能体配备完成任务所需的工具

    ### 任务设计
    1. **清晰的描述**: 详细说明任务要求
    2. **明确的输出**: 定义预期的输出格式和内容
    3. **合理的依赖**: 正确设置任务之间的依赖关系
    4. **适当的粒度**: 任务不要太大也不要太小

    ### 团队组织
    1. **专业分工**: 每个智能体专注于特定领域
    2. **流程选择**: 根据任务特点选择顺序或层级流程
    3. **规模控制**: 团队规模要适中,避免过度复杂
    4. **测试验证**: 充分测试团队协作效果
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 11. 性能优化

    ### 并行执行
    - 使用 `async_execution=True` 启用任务并行执行
    - 适用于相互独立的任务

    ### 缓存策略
    - 启用智能体缓存以减少重复计算
    - 使用工具缓存避免重复的API调用

    ### 资源管理
    - 合理设置 `max_iter` 限制迭代次数
    - 使用 `max_rpm` 控制API调用频率
    - 监控和优化token使用量
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 12. CrewAI vs Agno 框架对比

    以下是 CrewAI 和 Agno 两个主流智能体框架的对比:

    | 特性 | CrewAI | Agno |
    |------|--------|------|
    | **核心定位** | 多智能体协作框架 | 高性能 SDK 和运行时 |
    | **架构模式** | Crews (团队) + Flows (流程) | Agent + Team + Workflow |
    | **性能** | 良好 | 极致优化 (~3μs 实例化) |
    | **内存占用** | 中等 | 极低 (~6.5KB/Agent) |
    | **部署方式** | 需自行搭建 | 内置 AgentOS + FastAPI |
    | **UI 界面** | 无内置 UI | 内置管理 UI |
    | **数据隐私** | 依赖外部服务 | 完全私有部署 |
    | **工具生态** | 700+ 预构建工具 | 丰富的内置工具 + MCP 支持 |
    | **记忆系统** | 短期/长期/实体记忆 | 内置会话和知识管理 |
    | **学习曲线** | 中等 | 简单直观 |
    | **协作模式** | Sequential/Hierarchical | Team + Workflow |
    | **人机协作** | 支持 HITL | 内置 HITL |
    | **适用场景** | 复杂多智能体协作 | 高性能生产环境 |
    | **社区规模** | 10万+ 开发者 | 快速增长中 |
    | **开源协议** | MIT | Apache 2.0 |
    | **企业支持** | CrewAI AMP | AgentOS 云部署 |

    ### 选择建议

    **选择 CrewAI 如果你需要:**
    - 复杂的多智能体协作场景
    - 丰富的预构建工具集成
    - 成熟的社区和大量示例
    - 灵活的流程控制

    **选择 Agno 如果你需要:**
    - 极致的性能和低内存占用
    - 开箱即用的生产环境部署
    - 完全私有化的数据控制
    - 内置的管理和监控界面
    - 快速开发和上线

    ## 13. 总结

    CrewAI 是一个强大而灵活的多智能体框架,具有以下优势:

    - ✅ **独立框架**: 不依赖其他框架,性能优异
    - ✅ **灵活控制**: 支持高层编排和底层定制
    - ✅ **丰富工具**: 提供大量预构建工具
    - ✅ **活跃社区**: 超过10万认证开发者
    - ✅ **生产就绪**: 适用于企业级应用

    ### 下一步

    - 访问 [官方文档](https://docs.crewai.com) 了解更多
    - 查看 [示例仓库](https://github.com/crewAIInc/crewAI-examples) 获取灵感
    - 加入 [社区论坛](https://community.crewai.com) 交流讨论
    - 尝试 [CrewAI AMP](https://app.crewai.com) 企业版功能
    """
    )
    return


if __name__ == "__main__":
    app.run()

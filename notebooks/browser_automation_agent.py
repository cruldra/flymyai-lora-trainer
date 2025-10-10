import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 构建浏览器自动化代理

    浏览器仍然是最通用的界面，每天有43亿个页面被访问！

    今天，让我们演示如何使用本地技术栈完全自动化它：

    - [**Stagehand**](https://github.com/browserbase/stagehand-python) 开源AI浏览器自动化工具
    - CrewAI 用于编排
    - Ollama 运行 gpt-oss
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 系统概览

    ![系统架构](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0a709a88-8707-48d4-8c0d-b7438e1a1093_800x697.gif)

    工作流程：

    1. 用户输入自动化查询
    2. 规划代理创建自动化计划
    3. 浏览器自动化代理使用Stagehand工具执行计划
    4. 响应代理生成最终响应
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 定义LLM

    我们使用三个LLM：

    ![LLM架构](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3001dd24-b51f-4329-892b-5bee37eaa279_680x327.png)

    - **Planner LLM**: 为自动化任务创建结构化计划
    - **Automation LLM**: 使用Stagehand工具执行计划
    - **Response LLM**: 综合最终响应
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # 定义LLM配置示例
    llm_config = {
        "planner_llm": "用于创建结构化计划",
        "automation_llm": "用于执行浏览器自动化",
        "response_llm": "用于生成最终响应"
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 定义自动化规划代理

    规划代理接收用户的自动化任务，并为浏览器代理创建结构化的执行布局。

    ![规划代理](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F79fd954a-76be-4a30-a362-65d2706db762_680x538.png)
    """
    )
    return


@app.class_definition(hide_code=True)
# 自动化规划代理示例
class AutomationPlannerAgent:
    """
    接收用户任务并创建结构化执行计划
    """
    def __init__(self, llm):
        self.llm = llm

    def create_plan(self, task):
        """创建自动化任务的执行计划"""
        pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 定义Stagehand浏览器工具

    自定义CrewAI工具利用AI与网页交互。

    它利用Stagehand的计算机使用代理能力来自主导航URL、执行页面操作并提取数据以回答问题。

    ![Stagehand工具](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F196a789b-0757-4a99-ae5b-d157aeea01aa_679x560.png)
    """
    )
    return


@app.class_definition(hide_code=True)
# Stagehand浏览器工具示例
class StagehandBrowserTool:
    """
    使用AI进行网页交互的自定义CrewAI工具
    """
    def __init__(self):
        self.capabilities = [
            "自主导航URL",
            "执行页面操作",
            "提取数据"
        ]

    def navigate(self, url):
        """导航到指定URL"""
        pass

    def interact(self, action):
        """执行页面交互"""
        pass

    def extract_data(self, query):
        """提取数据以回答问题"""
        pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 定义浏览器自动化代理

    浏览器自动化代理利用前面提到的Stagehand工具进行自主浏览器控制和计划执行。

    ![浏览器自动化代理](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff88265a2-6fd1-4260-b0ae-5859e56f8466_679x513.png)
    """
    )
    return


@app.class_definition(hide_code=True)
# 浏览器自动化代理示例
class BrowserAutomationAgent:
    """
    使用Stagehand工具执行浏览器自动化任务
    """
    def __init__(self, llm, stagehand_tool):
        self.llm = llm
        self.stagehand_tool = stagehand_tool

    def execute_plan(self, plan):
        """执行自动化计划"""
        pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 定义响应综合代理

    综合代理作为最终的质量控制，精炼浏览器自动化代理的输出以生成优质响应。

    ![响应综合代理](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9141e6d6-1834-43ba-901a-dcae6c64056c_680x460.png)
    """
    )
    return


@app.class_definition(hide_code=True)
# 响应综合代理示例
class ResponseSynthesisAgent:
    """
    精炼浏览器自动化代理的输出并生成最终响应
    """
    def __init__(self, llm):
        self.llm = llm

    def synthesize_response(self, automation_output):
        """生成优质的最终响应"""
        pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 创建CrewAI代理流程

    最后，我们使用CrewAI Flows在工作流中连接我们的代理。

    ![CrewAI流程](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe166d013-9806-46fd-8d2f-da19c6c9b04f_680x564.png)
    """
    )
    return


@app.class_definition(hide_code=True)
# CrewAI工作流示例
class BrowserAutomationWorkflow:
    """
    多代理浏览器自动化工作流
    """
    def __init__(self, planner, automation_agent, response_agent):
        self.planner = planner
        self.automation_agent = automation_agent
        self.response_agent = response_agent

    def run(self, user_query):
        """
        执行完整的自动化工作流：
        1. 规划任务
        2. 执行浏览器自动化
        3. 综合响应
        """
        # 步骤1: 创建计划
        plan = self.planner.create_plan(user_query)

        # 步骤2: 执行计划
        automation_result = self.automation_agent.execute_plan(plan)

        # 步骤3: 生成最终响应
        final_response = self.response_agent.synthesize_response(automation_result)

        return final_response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 完成！

    这是我们的多代理浏览器自动化工作流的实际应用，我们要求它在Stagehand GitHub仓库中找到最大的贡献者：

    它启动了一个本地浏览器会话，导航到网页，并提取了信息。

    ## 相关资源

    - [Stagehand GitHub仓库](https://github.com/browserbase/stagehand-python)
    - [完整代码仓库](https://github.com/patchy631/ai-engineering-hub/tree/main/web-browsing-agent)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 总结

    本笔记本展示了如何构建一个完整的浏览器自动化代理系统，包括：

    - 使用多个专门的LLM处理不同任务
    - 规划代理负责任务分解
    - Stagehand工具提供浏览器自动化能力
    - 浏览器自动化代理执行具体操作
    - 响应综合代理确保输出质量
    - CrewAI Flows编排整个工作流

    这种架构可以应用于各种网页自动化场景，如数据抓取、自动化测试、信息收集等。
    """
    )
    return


if __name__ == "__main__":
    app.run()

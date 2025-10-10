import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Stagehand 完全指南")


@app.cell
def _():
    import marimo as mo
    import os
    from dotenv import load_dotenv
    import asyncio

    # 加载 .env 文件
    load_dotenv()

    # 检查环境变量
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY 已加载")
    else:
        print("⚠️ OPENAI_API_KEY 未设置")

    if os.getenv("OPENAI_API_BASE_URL"):
        print(f"✅ OPENAI_API_BASE_URL: {os.getenv('OPENAI_API_BASE_URL')}")

    if os.getenv("ANTHROPIC_API_KEY"):
        print(f"✅ ANTHROPIC_API_KEY: {os.getenv('ANTHROPIC_API_KEY')}")
    return mo, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🎭 Stagehand 完全指南

    ## 什么是 Stagehand?

    **Stagehand** 是一个 AI 驱动的浏览器自动化框架,由 Browserbase 团队开发。它结合了 AI 的灵活性和代码的精确性,让开发者能够可靠地自动化 Web 操作。

    ### 核心特性

    - 🎯 **精确控制**: 自由选择使用 AI 还是代码来执行操作
    - 🔄 **可重复执行**: 保存和重放操作,确保一致性
    - 🔧 **易于维护**: 一个脚本可以自动化多个网站,网站变化时自动适应
    - 🧩 **可组合工具**: 通过 Act、Extract、Observe 和 Agent 选择自动化级别
    - 🎨 **Playwright 兼容**: 完全兼容 Playwright API
    - 🐍 **双语言支持**: TypeScript 和 Python SDK

    ### 为什么选择 Stagehand?

    传统的浏览器自动化工具要么需要编写低级代码(Selenium、Playwright、Puppeteer),要么使用高级 AI 代理但在生产环境中不可预测。

    Stagehand 让你自由选择:
    - 在熟悉的页面使用代码获得精确控制
    - 在陌生的页面使用 AI 获得灵活性
    - 预览 AI 操作后再执行
    - 缓存可重复操作以节省时间和 token
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 1. 安装和配置

    ### 安装方式

    **快速开始 (推荐)**:
    ```bash
    npx create-browser-app
    ```

    **从源码构建**:
    ```bash
    git clone https://github.com/browserbase/stagehand.git
    cd stagehand
    pnpm install
    pnpm playwright install
    pnpm run build
    ```

    **Python 版本**:
    ```bash
    pip install stagehand
    ```

    ### 环境配置

    创建 `.env` 文件并添加 API 密钥:
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    BROWSERBASE_API_KEY=your_browserbase_api_key  # 可选,用于云端浏览器
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2. 四大核心原语

    Stagehand 提供四个强大的原语,让你精确控制 AI 的使用程度:

    ### Act - 执行自然语言操作
    使用自然语言执行单个操作,如点击、输入等。

    ### Extract - 提取结构化数据
    使用 schema 从页面中提取结构化数据。

    ### Observe - 发现可用操作
    发现页面上的可用操作和元素。

    ### Agent - 自主自动化工作流
    使用 Computer Use 模型自动化整个工作流程。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 3. Act - 执行操作

    `act()` 方法允许你使用自然语言执行单个操作:
    """
    )
    return


@app.cell
async def _(os):
    from stagehand import Stagehand
    import litellm
    print("=== Act 示例 ===")
    litellm._turn_on_debug()
    # 初始化 Stagehand (Python 版本使用驼峰命名)
    async with Stagehand(
        env="LOCAL",
        model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
        # modelClientOptions={
        #     "base_url": os.getenv("OPENAI_API_BASE_URL")
        # }
    ) as stagehand_act:
        page_act = stagehand_act.page

        # 导航到百度
        await page_act.goto("https://www.baidu.com")
        print("✅ 已导航到百度")

        # 使用自然语言操作
        await page_act.act("click on the search box")
        print("✅ 已点击搜索框")

        await page_act.act("type 'Stagehand AI automation'")
        print("✅ 已输入搜索文本")
    return (Stagehand,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4. Extract - 提取数据

    `extract()` 方法使用 schema 从页面中提取结构化数据:
    """
    )
    return


@app.cell
async def _(Stagehand, os):
    from pydantic import BaseModel, Field

    print("=== Extract 示例 ===")

    # 定义数据结构 (Python 版本需要 Pydantic 模型)
    class SearchInfo(BaseModel):
        query: str = Field(description="搜索查询")
        has_results: bool = Field(description="是否有搜索结果")

    # 初始化并提取数据
    async with Stagehand(
        env="LOCAL",
        model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
    ) as stagehand_extract:
        page_extract = stagehand_extract.page
        await page_extract.goto("https://www.baidu.com/s?wd=Stagehand")

        # 提取页面信息 (Python 版本返回 Pydantic 模型实例)
        result_extract = await page_extract.extract(
            instruction="extract the search query and whether there are results",
            schema=SearchInfo
        )

        print(f"搜索查询: {result_extract.query}")
        print(f"有结果: {result_extract.has_results}")
    return BaseModel, Field


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5. Observe - 发现操作

    `observe()` 方法帮助你发现页面上的可用操作:
    """
    )
    return


@app.cell
async def _(Stagehand, os):
    print("=== Observe 示例 ===")

    async with Stagehand(
        env="LOCAL",
       model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
    ) as stagehand_observe:
        page_observe = stagehand_observe.page
        await page_observe.goto("https://github.com/browserbase/stagehand")

        # 发现页面上的操作
        actions_observe = await page_observe.observe("find the star button")
        print(f"✅ 发现操作: {actions_observe}")

        # 发现导航元素
        nav_elements_observe = await page_observe.observe("find navigation menu items")
        print(f"✅ 发现导航元素: {nav_elements_observe}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 6. Agent - 自主工作流

    Agent 使用 Computer Use 模型自动化整个工作流程。

    **使用步骤**:
    1. 调用 `stagehand.agent()` 创建 agent 实例(可以指定模型、指令等参数)
    2. 调用 `agent.execute()` 执行任务

    **参数说明**:
    - `model`: 指定使用的模型,如果不指定则使用 Stagehand 的默认模型
    - `instructions`: 系统提示,告诉 agent 它的角色和行为
    - `options`: 额外选项,如 API 密钥等
    - `max_steps`: execute() 方法的最大步骤数
    - `auto_screenshot`: 是否自动截图
    """
    )
    return


@app.cell
async def _(Stagehand, os):
    print("=== Agent 示例 ===")

    # 在创建 Stagehand 实例时指定默认模型和密钥
    async with Stagehand(
        env="LOCAL",
        model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
    ) as stagehand_agent:
        await stagehand_agent.init()

        page_agent = stagehand_agent.page
        await page_agent.goto("https://github.com/browserbase/stagehand")

        # 1. 先调用 agent() 创建 agent 实例
        # 可以指定不同的模型,如果不指定则使用 Stagehand 的默认配置
        agent_demo = stagehand_agent.agent(
            model="claude-3-7-sonnet-latest",  # 可选,不指定则使用默认模型
            instructions="你是一个网页导航助手,帮助用户查找信息",  # 可选的系统提示
            options={"apiKey": os.getenv("ANTHROPIC_API_KEY")}  # 可选,不指定则使用默认密钥
        )

        # 2. 然后调用 execute() 执行任务
        result_agent = await agent_demo.execute(
            instruction="找到这个仓库的 star 数量",
            max_steps=10,  # 最大步骤数
            auto_screenshot=True  # 自动截图
        )

        print(f"✅ Agent 执行结果:")
        print(f"  完成: {result_agent.completed}")
        print(f"  消息: {result_agent.message}")
        print(f"  执行的操作数: {len(result_agent.actions) if result_agent.actions else 0}")
        if result_agent.usage:
            print(f"  使用情况: {result_agent.usage}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 7. 完整示例

    下面是一个完整的 GitHub 自动化示例:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    print("=== 完整 GitHub 自动化示例 ===")

    # 定义 PR 信息结构
    class PRInfo(BaseModel):
        title: str = Field(description="PR 标题")
        author: str = Field(description="PR 作者")

    async with Stagehand(
        env="LOCAL",
         model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
    ) as stagehand_github:
        page_github = stagehand_github.page

        # 使用 Playwright 函数导航
        await page_github.goto("https://github.com/browserbase/stagehand")
        print("✅ 已导航到 Stagehand 仓库")

        # 使用 act() 执行单个操作
        await page_github.act("click on the Pull requests tab")
        print("✅ 已点击 Pull requests 标签")

        # 等待页面加载
        await page_github.wait_for_timeout(2000)

        # 使用 extract() 提取第一个 PR 的数据
        pr_info_github = await page_github.extract(
            instruction="extract the title and author of the first pull request",
            schema=PRInfo
        )

        print(f"PR 标题: {pr_info_github.title}")
        print(f"PR 作者: {pr_info_github.author}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 8. 与 Playwright 集成

    Stagehand 完全兼容 Playwright API,你可以混合使用:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    print("=== Playwright 集成示例 ===")

    class PageTitle(BaseModel):
        title: str = Field(description="页面标题")

    async with Stagehand(
        env="LOCAL",
         model_name="anthropic/claude-3-7-sonnet-latest",
        model_api_key=os.getenv("ANTHROPIC_API_KEY")
    ) as stagehand_playwright:
        page_playwright = stagehand_playwright.page

        # 使用 Playwright 的精确导航
        await page_playwright.goto("https://github.com/browserbase/stagehand")
        print("✅ 使用 Playwright 导航")

        # 使用 Playwright 的等待功能
        await page_playwright.wait_for_load_state("networkidle")
        print("✅ 页面加载完成")

        # 使用 Stagehand 的 AI 能力提取数据
        page_info_playwright = await page_playwright.extract(
            instruction="extract the repository title",
            schema=PageTitle
        )

        print(f"仓库标题: {page_info_playwright.title}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 9. 缓存和重放

    Stagehand 支持缓存操作以提高性能和降低成本:
    """
    )
    return


@app.cell
async def _(Stagehand):
    print("=== 缓存示例 ===")

    # 启用缓存
    async with Stagehand(env="LOCAL", enable_caching=True) as stagehand_cache:
        page_cache = stagehand_cache.page
        await page_cache.goto("https://www.baidu.com")

        # 第一次执行会调用 LLM
        print("第一次执行 act...")
        await page_cache.act("click on the search box")
        print("✅ 第一次执行完成")

        # 刷新页面
        await page_cache.reload()

        # 第二次执行会使用缓存
        print("第二次执行 act (应该使用缓存)...")
        await page_cache.act("click on the search box")
        print("✅ 第二次执行完成")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 10. 调试和预览

    Stagehand 提供强大的调试功能:
    """
    )
    return


@app.cell
async def _(Stagehand):
    print("=== 调试模式示例 ===")

    # 启用调试模式 (显示浏览器窗口)
    async with Stagehand(env="LOCAL", headless=False, verbose=1) as stagehand_debug:
        page_debug = stagehand_debug.page
        await page_debug.goto("https://www.baidu.com")
        print("✅ 浏览器窗口已打开 (headless=False)")

        # 执行操作
        await page_debug.act("click on the search box")
        print("✅ 操作已执行")

        # 等待一下让用户看到浏览器
        await page_debug.wait_for_timeout(2000)

    print("✅ 调试模式演示完成")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 11. 实战示例 - 电商自动化

    下面是一个完整的电商网站自动化示例:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    from typing import List

    print("=== 电商自动化示例 ===")

    class ProductInfo(BaseModel):
        name: str = Field(description="产品名称")
        price: str = Field(description="产品价格")

    async with Stagehand(
        env="LOCAL",
        modelName="openai/gpt-4o-mini",
        modelApiKey=os.getenv("OPENAI_API_KEY"),
        modelClientOptions={
            "base_url": os.getenv("OPENAI_API_BASE_URL")
        }
    ) as stagehand_ecommerce:
        page_ecommerce = stagehand_ecommerce.page

        # 导航到亚马逊
        await page_ecommerce.goto("https://www.amazon.com")
        print("✅ 已导航到 Amazon")

        # 搜索产品
        await page_ecommerce.act("search for 'laptop'")
        print("✅ 已搜索 laptop")

        # 等待结果加载
        await page_ecommerce.wait_for_timeout(3000)

        # 提取第一个产品信息
        product_ecommerce = await page_ecommerce.extract(
            instruction="extract the name and price of the first product",
            schema=ProductInfo
        )

        print(f"产品名称: {product_ecommerce.name}")
        print(f"产品价格: {product_ecommerce.price}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 12. 实战示例 - 社交媒体自动化

    自动化社交媒体操作:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    print("=== 社交媒体自动化示例 ===")

    class RepoInfo(BaseModel):
        name: str = Field(description="仓库名称")
        stars: str = Field(description="star 数量")
        description: str = Field(description="仓库描述")

    async with Stagehand(
        env="LOCAL",
        modelName="openai/gpt-4o-mini",
        modelApiKey=os.getenv("OPENAI_API_KEY"),
        modelClientOptions={
            "base_url": os.getenv("OPENAI_API_BASE_URL")
        }
    ) as stagehand_social:
        page_social = stagehand_social.page

        # 导航到 GitHub trending
        await page_social.goto("https://github.com/trending")
        print("✅ 已导航到 GitHub Trending")

        # 等待页面加载
        await page_social.wait_for_timeout(2000)

        # 提取第一个热门仓库信息
        repo_social = await page_social.extract(
            instruction="extract the name, stars, and description of the first trending repository",
            schema=RepoInfo
        )

        print(f"仓库名称: {repo_social.name}")
        print(f"Star 数量: {repo_social.stars}")
        print(f"描述: {repo_social.description}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 13. 高级特性

    ### 多页面管理

    Stagehand 支持同时管理多个页面:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    print("=== 多页面管理示例 ===")

    class SiteInfo(BaseModel):
        title: str = Field(description="网站标题")

    async with Stagehand(
        env="LOCAL",
        modelName="openai/gpt-4o-mini",
        modelApiKey=os.getenv("OPENAI_API_KEY"),
        modelClientOptions={
            "base_url": os.getenv("OPENAI_API_BASE_URL")
        }
    ) as stagehand_multipage:
        # 获取主页面
        page_multipage = stagehand_multipage.page

        # 在不同页面执行操作
        await page_multipage.goto("https://github.com/browserbase/stagehand")
        print("✅ Page 1: 已导航到 Stagehand")

        # 提取数据
        info_multipage = await page_multipage.extract(
            instruction="extract the page title",
            schema=SiteInfo
        )

        print(f"Page 1 标题: {info_multipage.title}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### 错误处理和重试

    处理网络错误和元素未找到的情况:
    """
    )
    return


@app.cell
async def _(BaseModel, Field, Stagehand, os):
    print("=== 错误处理示例 ===")

    class PageData(BaseModel):
        title: str = Field(description="页面标题")

    async with Stagehand(
        env="LOCAL",
        modelName="openai/gpt-4o-mini",
        modelApiKey=os.getenv("OPENAI_API_KEY"),
        modelClientOptions={
            "base_url": os.getenv("OPENAI_API_BASE_URL")
        }
    ) as stagehand_error:
        page_error = stagehand_error.page

        try:
            await page_error.goto("https://github.com/browserbase/stagehand")
            print("✅ 页面加载成功")

            # 提取数据
            data_error = await page_error.extract(
                instruction="extract the page title",
                schema=PageData
            )
            print(f"提取成功: {data_error.title}")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            data_error = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 14. 最佳实践

    ### 何时使用 AI vs 代码

    - ✅ **使用 AI (act/extract)** 当:
      - 页面结构不确定或经常变化
      - 需要理解页面语义
      - 元素没有稳定的选择器
      - 需要处理多种页面布局

    - ✅ **使用代码 (Playwright)** 当:
      - 页面结构已知且稳定
      - 需要最快的执行速度
      - 需要精确控制
      - 元素有稳定的 ID 或选择器

    ### 性能优化

    - 启用缓存以减少 LLM 调用
    - 使用 `headless: true` 提高速度
    - 批量提取数据而不是逐个提取
    - 使用 Playwright 的等待策略避免不必要的延迟

    ### 可靠性建议

    - 始终设置超时
    - 实现错误处理和重试逻辑
    - 使用预览模式验证操作
    - 记录详细日志以便调试
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 15. 总结

    Stagehand 是一个强大的 AI 浏览器自动化框架,它结合了:

    - ✅ AI 的灵活性和适应性
    - ✅ 代码的精确性和可靠性
    - ✅ Playwright 的完整功能
    - ✅ 易于使用的 API
    - ✅ 生产级的性能和稳定性

    ### 相关资源

    - 📚 [官方文档](https://docs.stagehand.dev)
    - 💻 [GitHub 仓库](https://github.com/browserbase/stagehand)
    - 🐍 [Python SDK](https://github.com/browserbase/stagehand-python)
    - 💬 [Slack 社区](https://join.slack.com/t/stagehand-dev/shared_invite/zt-38khc8iv5-T2acb50_0OILUaX7lxeBOg)
    - 🎬 [快速开始视频](https://www.loom.com/share/f5107f86d8c94fa0a8b4b1e89740f7a7)
    """
    )
    return


if __name__ == "__main__":
    app.run()

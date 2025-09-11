import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import argparse
    import sys
    from io import StringIO
    import contextlib

    mo.md("""
    # argparse 模块完整教程

    argparse 是 Python 标准库中用于解析命令行参数的强大模块。本教程将详细介绍 argparse 的各个 API 和用法。
    """)
    return StringIO, argparse, mo, sys


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. ArgumentParser 类

    ArgumentParser 是 argparse 模块的核心类，用于创建命令行参数解析器。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # ArgumentParser 基本用法
    parser = argparse.ArgumentParser(
        prog='my_program',
        description='这是一个示例程序',
        epilog='更多信息请访问官方文档',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
        allow_abbrev=True
    )

    mo.md(f"""
    ### ArgumentParser 构造函数参数：

    - **prog**: 程序名称 (默认: sys.argv[0])
    - **description**: 程序描述
    - **epilog**: 帮助信息末尾显示的文本
    - **formatter_class**: 帮助信息格式化类
    - **add_help**: 是否自动添加 -h/--help 选项
    - **allow_abbrev**: 是否允许长选项缩写

    创建的解析器: {parser}
    """)
    return (parser,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. add_argument() 方法

    add_argument() 是添加命令行参数的核心方法。
    """
    )
    return


@app.cell
def _(mo, parser):
    # 位置参数
    parser.add_argument('filename', help='要处理的文件名')

    # 可选参数
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('-n', '--number', type=int, default=10, help='数字参数')

    # 选择参数
    parser.add_argument('--mode', choices=['read', 'write', 'append'], default='read', help='操作模式')

    # 多值参数
    parser.add_argument('--files', nargs='+', help='多个文件')
    parser.add_argument('--coords', nargs=2, type=float, help='坐标 (x, y)')

    mo.md("""
    ### add_argument() 常用参数：

    - **name or flags**: 参数名称或标志
    - **action**: 参数动作 ('store', 'store_true', 'store_false', 'append', 'count' 等)
    - **nargs**: 参数个数 ('?', '*', '+', 数字)
    - **const**: action 和 nargs 需要的常量值
    - **default**: 默认值
    - **type**: 参数类型转换函数
    - **choices**: 允许的值列表
    - **required**: 是否必需 (仅适用于可选参数)
    - **help**: 帮助信息
    - **metavar**: 帮助信息中显示的参数名
    - **dest**: 解析后存储的属性名
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. action 参数详解

    action 参数定义了参数被解析时的行为。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 创建新的解析器演示 action
    action_parser = argparse.ArgumentParser(description='Action 示例')

    # store (默认)
    action_parser.add_argument('--name', action='store', help='存储值')

    # store_true/store_false
    action_parser.add_argument('--enable', action='store_true', help='设置为 True')
    action_parser.add_argument('--disable', action='store_false', help='设置为 False')

    # store_const
    action_parser.add_argument('--const-val', action='store_const', const=42, help='存储常量')

    # append
    action_parser.add_argument('--item', action='append', help='添加到列表')

    # count
    action_parser.add_argument('-v', '--verbose', action='count', default=0, help='详细级别')

    # version
    action_parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    mo.md("""
    ### Action 类型：

    - **store**: 存储参数值 (默认)
    - **store_true**: 存储 True
    - **store_false**: 存储 False  
    - **store_const**: 存储常量值
    - **append**: 将值添加到列表
    - **append_const**: 将常量添加到列表
    - **count**: 计算参数出现次数
    - **help**: 显示帮助信息并退出
    - **version**: 显示版本信息并退出
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. nargs 参数详解

    nargs 参数控制参数可以接受的值的数量。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 创建新的解析器演示 nargs
    nargs_parser = argparse.ArgumentParser(description='nargs 示例')

    # 固定数量
    nargs_parser.add_argument('--coords', nargs=2, type=float, help='两个坐标值')
    nargs_parser.add_argument('--rgb', nargs=3, type=int, help='RGB 颜色值')

    # 可选数量
    nargs_parser.add_argument('--optional', nargs='?', const='default', help='可选参数')
    nargs_parser.add_argument('--zero-or-more', nargs='*', help='零个或多个参数')
    nargs_parser.add_argument('--one-or-more', nargs='+', help='一个或多个参数')

    # 剩余所有参数
    nargs_parser.add_argument('files', nargs='*', help='文件列表')

    mo.md("""
    ### nargs 值：

    - **数字**: 精确的参数个数
    - **'?'**: 0 或 1 个参数
    - **'*'**: 0 或多个参数
    - **'+'**: 1 或多个参数
    - **argparse.REMAINDER**: 剩余所有参数
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. 类型转换和验证

    type 参数用于将字符串参数转换为其他类型。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 类型转换示例
    type_parser = argparse.ArgumentParser(description='类型转换示例')

    # 基本类型
    type_parser.add_argument('--int-val', type=int, help='整数')
    type_parser.add_argument('--float-val', type=float, help='浮点数')
    type_parser.add_argument('--bool-val', type=bool, help='布尔值')

    # 文件类型
    type_parser.add_argument('--input-file', type=argparse.FileType('r'), help='输入文件')
    type_parser.add_argument('--output-file', type=argparse.FileType('w'), help='输出文件')

    # 自定义类型函数
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} 不是正整数")
        return ivalue

    type_parser.add_argument('--positive', type=positive_int, help='正整数')

    mo.md("""
    ### 类型转换：

    - **内置类型**: int, float, str, bool
    - **argparse.FileType**: 文件类型，自动打开文件
    - **自定义函数**: 可以定义自己的类型转换和验证函数

    自定义类型函数应该：
    - 接受一个字符串参数
    - 返回转换后的值
    - 在无效时抛出 ArgumentTypeError
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. 参数组 (Argument Groups)

    参数组用于在帮助信息中组织相关的参数。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 参数组示例
    group_parser = argparse.ArgumentParser(description='参数组示例')

    # 创建参数组
    input_group = group_parser.add_argument_group('输入选项', '控制输入的参数')
    input_group.add_argument('--input-file', help='输入文件')
    input_group.add_argument('--input-format', choices=['json', 'xml', 'csv'], help='输入格式')

    output_group = group_parser.add_argument_group('输出选项', '控制输出的参数')
    output_group.add_argument('--output-file', help='输出文件')
    output_group.add_argument('--output-format', choices=['json', 'xml', 'csv'], help='输出格式')

    # 互斥参数组
    mutex_group = group_parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument('--create', action='store_true', help='创建新文件')
    mutex_group.add_argument('--update', action='store_true', help='更新现有文件')
    mutex_group.add_argument('--delete', action='store_true', help='删除文件')

    mo.md("""
    ### 参数组类型：

    - **add_argument_group()**: 普通参数组，用于组织帮助信息
    - **add_mutually_exclusive_group()**: 互斥参数组，只能选择其中一个参数

    互斥组的 required 参数：
    - True: 必须选择组中的一个参数
    - False: 可以不选择任何参数
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. 子命令 (Subparsers)

    子命令允许程序支持多个不同的操作模式。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 子命令示例
    main_parser = argparse.ArgumentParser(description='主程序')
    main_parser.add_argument('--verbose', action='store_true', help='详细输出')

    # 创建子解析器
    subparsers = main_parser.add_subparsers(
        dest='command',
        help='可用命令',
        title='子命令',
        description='支持的操作'
    )

    # 创建 'create' 子命令
    create_parser = subparsers.add_parser('create', help='创建新项目')
    create_parser.add_argument('name', help='项目名称')
    create_parser.add_argument('--template', help='项目模板')

    # 创建 'build' 子命令
    build_parser = subparsers.add_parser('build', help='构建项目')
    build_parser.add_argument('--target', default='debug', choices=['debug', 'release'], help='构建目标')
    build_parser.add_argument('--jobs', type=int, default=1, help='并行任务数')

    # 创建 'test' 子命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('--pattern', help='测试文件模式')
    test_parser.add_argument('--coverage', action='store_true', help='生成覆盖率报告')

    mo.md("""
    ### 子命令特性：

    - **dest**: 存储选择的子命令名称的属性
    - **help**: 子命令的帮助信息
    - **title**: 子命令组的标题
    - **description**: 子命令组的描述

    每个子解析器都是独立的 ArgumentParser 实例，可以有自己的参数。
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. 解析参数

    使用 parse_args() 方法解析命令行参数。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 解析参数示例
    demo_parser = argparse.ArgumentParser(description='解析示例')
    demo_parser.add_argument('name', help='名称')
    demo_parser.add_argument('--age', type=int, default=18, help='年龄')
    demo_parser.add_argument('--verbose', action='store_true', help='详细输出')
    demo_parser.add_argument('--tags', nargs='*', help='标签列表')

    # 模拟命令行参数
    test_args = ['张三', '--age', '25', '--verbose', '--tags', 'python', 'programming']

    # 解析参数
    args = demo_parser.parse_args(test_args)

    mo.md(f"""
    ### 解析结果：

    ```python
    # 模拟命令行: python script.py {' '.join(test_args)}
    args = parser.parse_args({test_args})

    # 解析结果:
    args.name = {args.name!r}
    args.age = {args.age!r}
    args.verbose = {args.verbose!r}
    args.tags = {args.tags!r}
    ```

    ### 其他解析方法：

    - **parse_args(args=None)**: 解析参数列表 (默认 sys.argv[1:])
    - **parse_known_args()**: 解析已知参数，返回 (namespace, remaining_args)
    - **parse_intermixed_args()**: 解析混合的位置和可选参数
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 9. 错误处理

    argparse 提供了多种错误处理机制。
    """
    )
    return


@app.cell
def _(StringIO, argparse, mo, sys):
    # 错误处理示例
    error_parser = argparse.ArgumentParser(description='错误处理示例')
    error_parser.add_argument('--number', type=int, required=True, help='必需的数字')
    error_parser.add_argument('--choice', choices=['a', 'b', 'c'], help='选择项')

    # 捕获解析错误
    def safe_parse(parser, args_list):
        try:
            return parser.parse_args(args_list)
        except SystemExit as e:
            return f"解析失败，退出码: {e.code}"

    # 测试各种错误情况
    error_cases = [
        [],  # 缺少必需参数
        ['--number', 'abc'],  # 类型错误
        ['--number', '123', '--choice', 'x'],  # 选择错误
        ['--unknown', 'value']  # 未知参数
    ]

    error_results = []
    for case in error_cases:
        # 重定向 stderr 来捕获错误信息
        old_stderr = sys.stderr
        sys.stderr = captured_output = StringIO()

        parse_result = safe_parse(error_parser, case)
        error_msg = captured_output.getvalue()

        sys.stderr = old_stderr

        error_results.append((case, parse_result, error_msg.strip()))

    mo.md(f"""
    ### 错误处理结果：

    {chr(10).join([f"**参数**: {case or '(空)'}<br>**结果**: {parse_result}<br>**错误信息**: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}<br>" for case, parse_result, error_msg in error_results])}

    ### 自定义错误处理：

    可以通过继承 ArgumentParser 并重写 error() 方法来自定义错误处理。
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 10. 高级特性

    argparse 还提供了一些高级特性。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 自定义 Action 类
    class CustomAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # 自定义处理逻辑
            processed_value = f"处理后的值: {values.upper()}"
            setattr(namespace, self.dest, processed_value)

    # 使用自定义 Action
    advanced_parser = argparse.ArgumentParser(description='高级特性示例')
    advanced_parser.add_argument('--custom', action=CustomAction, help='使用自定义 Action')

    # 配置文件支持
    advanced_parser.add_argument('--config', help='配置文件路径')

    # 环境变量默认值
    import os
    advanced_parser.add_argument('--env-var', default=os.environ.get('MY_VAR', 'default'), help='从环境变量获取默认值')

    # 自定义帮助格式
    class CustomHelpFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings) + ' ' + args_string

    custom_formatter_parser = argparse.ArgumentParser(
        description='自定义格式示例',
        formatter_class=CustomHelpFormatter
    )
    custom_formatter_parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    mo.md("""
    ### 高级特性：

    1. **自定义 Action 类**：
       - 继承 argparse.Action
       - 实现 __call__ 方法
       - 可以进行复杂的参数处理

    2. **环境变量集成**：
       - 使用 os.environ.get() 设置默认值
       - 可以从环境变量读取配置

    3. **自定义帮助格式**：
       - 继承 HelpFormatter 类
       - 重写格式化方法
       - 自定义帮助信息显示
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 11. 实际应用示例

    让我们创建一个完整的命令行工具示例。
    """
    )
    return


@app.cell
def _(argparse, mo):
    # 完整的命令行工具示例
    def create_file_processor():
        parser = argparse.ArgumentParser(
            prog='file-processor',
            description='文件处理工具',
            epilog='示例: file-processor process input.txt --output output.txt --format json'
        )

        # 全局选项
        parser.add_argument('-v', '--verbose', action='count', default=0, help='详细级别 (-v, -vv, -vvv)')
        parser.add_argument('--config', help='配置文件路径')

        # 子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')

        # process 子命令
        process_parser = subparsers.add_parser('process', help='处理文件')
        process_parser.add_argument('input_file', help='输入文件')
        process_parser.add_argument('-o', '--output', help='输出文件')
        process_parser.add_argument('--format', choices=['json', 'xml', 'csv'], default='json', help='输出格式')
        process_parser.add_argument('--encoding', default='utf-8', help='文件编码')

        # validate 子命令
        validate_parser = subparsers.add_parser('validate', help='验证文件')
        validate_parser.add_argument('files', nargs='+', help='要验证的文件')
        validate_parser.add_argument('--schema', help='验证模式文件')
        validate_parser.add_argument('--strict', action='store_true', help='严格模式')

        # convert 子命令
        convert_parser = subparsers.add_parser('convert', help='转换文件格式')
        convert_parser.add_argument('input_file', help='输入文件')
        convert_parser.add_argument('output_file', help='输出文件')

        # 转换选项组
        convert_group = convert_parser.add_argument_group('转换选项')
        convert_group.add_argument('--from-format', required=True, choices=['json', 'xml', 'csv'], help='源格式')
        convert_group.add_argument('--to-format', required=True, choices=['json', 'xml', 'csv'], help='目标格式')
        convert_group.add_argument('--preserve-order', action='store_true', help='保持字段顺序')

        return parser

    # 创建解析器
    file_processor = create_file_processor()

    # 测试不同的命令
    test_commands = [
        ['process', 'input.txt', '--output', 'output.json', '--format', 'json', '-vv'],
        ['validate', 'file1.json', 'file2.json', '--schema', 'schema.json', '--strict'],
        ['convert', 'data.csv', 'data.json', '--from-format', 'csv', '--to-format', 'json']
    ]

    command_results = []
    for cmd in test_commands:
        try:
            cmd_result = file_processor.parse_args(cmd)
            command_results.append((cmd, cmd_result))
        except SystemExit:
            command_results.append((cmd, "解析失败"))

    mo.md(f"""
    ### 文件处理工具示例：

    {chr(10).join([f"**命令**: `{' '.join(cmd)}`<br>**解析结果**: {cmd_result}<br>" for cmd, cmd_result in command_results])}

    这个示例展示了：
    - 多个子命令
    - 不同类型的参数
    - 参数组织
    - 实际的命令行工具结构
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 12. 最佳实践

    使用 argparse 的一些最佳实践建议。
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### argparse 最佳实践：

    1. **清晰的帮助信息**：
       - 为每个参数提供有意义的 help 文本
       - 使用 description 和 epilog 提供程序概述
       - 选择合适的 metavar 让帮助信息更清晰

    2. **合理的默认值**：
       - 为可选参数设置合理的默认值
       - 考虑从环境变量或配置文件读取默认值
       - 使用 required=True 标记必需的可选参数

    3. **类型验证**：
       - 使用 type 参数进行类型转换
       - 编写自定义类型函数进行复杂验证
       - 使用 choices 限制可选值

    4. **错误处理**：
       - 提供清晰的错误信息
       - 考虑使用 parse_known_args() 处理部分解析
       - 自定义错误处理逻辑

    5. **代码组织**：
       - 将解析器创建封装在函数中
       - 使用参数组组织相关参数
       - 对于复杂工具使用子命令

    6. **测试**：
       - 编写单元测试验证参数解析
       - 测试各种错误情况
       - 验证帮助信息的正确性
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 13. 总结

    argparse 是一个功能强大且灵活的命令行参数解析库，提供了：

    - **ArgumentParser**: 核心解析器类
    - **add_argument()**: 添加各种类型的参数
    - **Action 系统**: 灵活的参数处理机制
    - **类型转换**: 自动类型转换和验证
    - **参数组**: 组织和管理相关参数
    - **子命令**: 支持复杂的命令行工具
    - **错误处理**: 完善的错误处理机制
    - **自定义扩展**: 支持自定义 Action 和格式化器

    通过合理使用这些特性，可以创建出专业、易用的命令行工具。
    """
    )
    return


if __name__ == "__main__":
    app.run()

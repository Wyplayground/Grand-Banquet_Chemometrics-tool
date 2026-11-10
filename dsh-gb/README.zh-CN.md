# Grand Banquet for DeepSeek Harness

[English](README.md) · 简体中文

**公开版 v0.0** · DeepSeek Harness Web 0.2.2-rc.1 · Python 3.10–3.13

在 Harness 工作台、GB 分析对话或本地 MCP 客户端运行 Grand Banquet（GB）化学计量分析。插件支持 CSV 和 Excel 数据、分类与回归、候选模型比较、模型导出和新样本预测。

## 安装到 Harness

在运行 Harness 的电脑上安装 Git 和 Python。从 [Harness 插件发布页](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool/releases/tag/dsh-gb-v0.0)下载安装包，再装入 Web 配置。

```powershell
dsh plugin --profile web add ./dsh-gb-0.0.0.tgz
dsh web
```

如果 Harness 已在运行，安装后先重启。打开 **插件 → GB 工作台**，选择分析工作区，点击 **创建独立环境并安装 GB**。安装完成后点击 **使用此环境并检查**。安装器会获取锁定版本的 GB 源码，并把通过哈希校验的 Python 依赖安装到工作区下的 `.dsh-gb/runtime/`。

## 运行分析

1. 选择 CSV、XLSX 或 XLS 文件。Excel 文件需选择工作表。映射目标列、数值特征列，以及可选的样本编号、分组和划分列，确认后生成分析副本。
2. 选择评估模式和搜索范围。自定义搜索最多 32 个候选，也可以为当前任务选择 **全部 990 候选**。确认计划后开始运行。任务中断后可以从已保存的检查点继续。
3. 比较候选指标、查看诊断图并下载结果。导出选定模型后，选择新样本文件并匹配训练特征列，即可生成预测 CSV。

在新的 Harness 对话中选择 **GB 分析**，可以让助手检查数据、在运行前展示计划，并呈现摘要、结果表和诊断图。对话使用现有的 Harness 模型登录。工作台直接调用本地 GB 引擎。Harness 工具调用包含文件路径、分析设置、数据概况和结果摘要，原始数据保存在工作区。

GB 使用交叉验证指标（`CV_*`）对候选排序。`Val_*` 是内部留出集指标。[GB 评估方法](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool/blob/main/Evaluation%20methods.md)

## 接入本地 MCP 客户端

克隆对应版本并安装 Node 和 Python 依赖。在 MCP 客户端添加本地 stdio 服务，将示例中的路径替换为仓库位置和安装器输出的路径。

```powershell
git clone --branch dsh-gb-v0.0 https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool.git
cd Grand-Banquet_Chemometrics-tool/dsh-gb
npm ci --ignore-scripts
python scripts/install.py --workspace "D:/GB-analysis" --install-deps
```

```json
{
  "mcpServers": {
    "gb": {
      "command": "node",
      "args": [
        "D:/tools/dsh-gb/mcp.js",
        "--python", "D:/GB-analysis/.dsh-gb/runtime/venv/Scripts/python.exe",
        "--gb-root", "D:/GB-analysis/.dsh-gb/runtime/gb",
        "--workspace", "D:/GB-analysis",
        "--scope", "research"
      ]
    }
  }
}
```

Linux 和 macOS 的独立解释器位于 `venv/bin/python`。先调用 `gb_environment` 检查环境。将 `gb_plan_analysis` 的 `library` 设为 `"full"` 即可规划全部 990 个候选。调用 `gb_run_analysis` 后，用 `gb_job_status` 查询返回的任务。分析结果、导出模型和预测按 MCP scope 分开保存。

[Apache-2.0](LICENSE) · [Grand Banquet](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool)

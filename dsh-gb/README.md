# Grand Banquet for DeepSeek Harness

English · [Simplified Chinese](README.zh-CN.md)

**Public release v0.0** · DeepSeek Harness Web 0.2.2-rc.1 · Python 3.10–3.13

Run Grand Banquet (GB) chemometrics from the Harness workbench, a GB analysis conversation, or a local MCP client. The plugin supports CSV and Excel data, classification and regression, candidate comparison, model export, and prediction.

## Install in Harness

Install Git and Python on the machine running Harness. Download the package from the [Harness plugin release](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool/releases/tag/dsh-gb-v0.0), then install it into the Web profile.

```powershell
dsh plugin --profile web add ./dsh-gb-0.0.0.tgz
dsh web
```

Restart Harness if it was already running. Open the GB workbench from the Plugins page and choose an analysis workspace. Use the setup button to create a private environment and install GB. When setup finishes, apply the environment and run the environment check. The installer fetches the pinned GB source and installs hash-verified Python wheels under `<workspace>/.dsh-gb/runtime/`.

The workbench controls are displayed in Chinese. See the exact button labels in the [Simplified Chinese guide](README.zh-CN.md).

## Run an analysis

1. Select a CSV, XLSX, or XLS file and, for Excel, its sheet. Map the target, numeric predictors, and any sample ID, group, or split columns. Confirm the mapping to create a working copy.
2. Choose an evaluation mode and either a custom search of up to 32 candidates or the full library of 990 candidates for the selected task. Review the plan and start it. Interrupted runs can resume from saved checkpoints.
3. Compare candidate metrics, view diagnostic plots, and download results. Export a chosen model, select a new data file, and map its columns to the training features to generate a prediction CSV.

In a new Harness conversation, select the GB analysis preset to use GB tools. Ask the assistant to inspect the data, show its plan before execution, and present the summary, results table, and plots. Conversations use your Harness model login. The workbench runs directly on the local GB engine. Harness tool calls include file paths, settings, metadata, and result summaries. Source data stays in the workspace.

GB orders candidates by cross-validation metrics (`CV_*`). Metrics labeled `Val_*` describe the internal holdout. [GB evaluation methods](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool/blob/main/Evaluation%20methods.md)

## Connect a local MCP client

Clone this release and install its Node and Python dependencies. Add a local stdio server to your MCP client using the example configuration. Substitute the repository location and the paths printed by the installer.

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

On Linux and macOS, the private interpreter is under `venv/bin/python`. Start with `gb_environment`. Use `gb_plan_analysis` with `library: "full"` to plan all 990 candidates. Check the job returned by `gb_run_analysis` with `gb_job_status`. Each MCP scope stores its own results, model exports, and predictions.

[Apache-2.0](LICENSE) · [Grand Banquet](https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool)

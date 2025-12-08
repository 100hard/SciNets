# VS Code settings for SciNetsV2

You can create `.vscode/launch.json` manually with the following content for debugging:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Server (Port 8000)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
      "cwd": "${workspaceFolder}/backend",
      "env": {"PYTHONPATH": "${workspaceFolder}/backend"},
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Debug Current Test File",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v", "-s"],
      "cwd": "${workspaceFolder}/backend",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

This file is not tracked in git to avoid conflicts.

# Setup Guide for multiagentz

This guide is designed to be followed by Claude Code or any AI assistant.
Give these instructions to Claude Code and it will walk you through setup.

## Instructions for Claude Code

```
Install, set up, and configure the multiagentz Python package located in this
directory. Walk me through everything step by step using multiple choice
questions whenever you need input from me.

Here's what needs to happen:
1. Check if Python 3.10+ and pip are installed (install via Homebrew if not)
2. Create a virtual environment and install the package
3. Ask me which AI providers I want to use (Anthropic, OpenAI, xAI, Google)
4. For each provider I choose, ask me to paste my API key
5. Update multiagentz/providers.py with my keys
6. Help me create my first YAML stack config by asking about my project
7. Test that everything works by running: maz --config stacks/example.yaml

Be friendly and use multiple choice questions whenever possible.
Don't assume I know programming terminology — explain things simply.
```

## Manual Setup (if not using Claude Code)

### 1. Prerequisites

```bash
# Check Python version (need 3.10+)
python3 --version

# If not installed, use Homebrew:
brew install python@3.12
```

### 2. Install

```bash
# Navigate to the package directory
cd /path/to/multiagentz-pkg

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all provider support
pip install -e ".[all]"
```

### 3. Add API Keys

Edit `multiagentz/providers.py` and replace the placeholder keys:

```python
"anthropic": {
    "api_key": "sk-ant-YOUR-KEY-HERE",
    ...
},
```

### 4. Run

```bash
# Activate the virtual environment (if not already)
source .venv/bin/activate

# Run with example config
maz --config stacks/example.yaml

# Or create your own stack config based on stacks/template.yaml
```

### 5. Create Your Stack

Copy `stacks/template.yaml` to `stacks/my-project.yaml` and customize:
- Update `repo_path` to point to your code directories
- Update `key_files` to your important source files
- Adjust descriptions and system prompts
- Optionally set per-agent models

Then run:
```bash
maz --config stacks/my-project.yaml
```

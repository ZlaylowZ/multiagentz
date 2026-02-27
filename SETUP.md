# Getting Started with multiagentz

Everything below is designed for someone who has never touched a terminal before.
Follow the steps in order. If you get stuck, paste the error into Claude Code and it will help.

---

## Step 1: Install Claude Code (your AI assistant)

Claude Code is a terminal-based AI assistant that will handle all the technical
setup for you. You just talk to it and it does the work.

### 1a. Install Claude Code Desktop App

1. Open Safari and go to: **https://claude.ai/download**
2. Click **"Download for Mac"**
3. Open the downloaded `.dmg` file
4. Drag **Claude** into your **Applications** folder
5. Open Claude from your Applications (or Spotlight: Cmd+Space, type "Claude")
6. Sign in with your Anthropic account (or create one at https://claude.ai)

### 1b. Enable Claude Code (the terminal tool)

Claude Code is a feature inside the Claude desktop app that lets Claude
read files, run commands, and write code on your computer.

1. Open the Claude desktop app
2. Go to **Settings** (gear icon, top-right) → **Developer**
3. Look for **"Claude Code"** or **"Computer Use"** and enable it
4. If prompted, grant Terminal/file access permissions

> **Alternative: Install Claude Code via terminal (if you prefer)**
>
> If you're comfortable with terminal, you can also install Claude Code as
> a standalone CLI tool:
>
> 1. Open **Terminal** (Spotlight: Cmd+Space, type "Terminal")
> 2. Run: `npm install -g @anthropic-ai/claude-code`
>    - If `npm` isn't found, install Node.js first: `brew install node`
>    - If `brew` isn't found, see Step 1c below
> 3. Run: `claude` to start a session
> 4. Authenticate when prompted

### 1c. Install Homebrew (Mac's package manager — you'll need this)

Homebrew installs developer tools on your Mac. You almost certainly need it.

1. Open **Terminal** (Spotlight: Cmd+Space, type "Terminal")
2. Paste this entire line and press Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. It will ask for your Mac password (the one you use to log in). Type it — nothing will appear as you type, that's normal. Press Enter.
4. Wait 2-5 minutes for it to install.
5. **Important:** Follow any instructions it prints at the end about adding Homebrew to your PATH. It usually looks like:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

6. Verify it works: `brew --version`

---

## Step 2: Get your AI provider API keys

You need at least **one** API key to use multiagentz. More providers = more model options.

| Provider | Where to get key | Models you get |
|----------|-----------------|----------------|
| **Anthropic** | https://console.anthropic.com/settings/keys | Claude Opus 4, Sonnet 4, Haiku 3.5 |
| **OpenAI** | https://platform.openai.com/api-keys | GPT-4o, o4-mini |
| **xAI** | https://console.x.ai/ | Grok-4, Grok-4.1 |
| **Google** | https://aistudio.google.com/apikey | Gemini 2.0 Flash, 2.5 Pro |

For each provider you want:
1. Visit the link above
2. Create an account if needed
3. Generate an API key
4. **Save it somewhere** — you'll paste it in during setup

> **Recommendation:** Start with just **Anthropic** (Claude). You can add more later.

---

## Step 3: Let Claude Code do the rest

Once Claude Code is running (either the desktop app or CLI), navigate to
your multiagentz folder and give it these instructions:

### If using Claude Code CLI:

```bash
cd /path/to/multiagentz-pkg
claude
```

### Then paste this prompt:

```
Install, set up, and configure the multiagentz Python package in this directory.
Walk me through everything step by step. Use multiple choice questions whenever
you need input from me.

Here's what needs to happen:
1. Check if Python 3.10+ is installed (install via Homebrew if needed)
2. Create a Python virtual environment and install the package with: pip install -e ".[all]"
3. Ask me which AI providers I want to use (Anthropic, OpenAI, xAI, Google)
4. For each provider I choose, ask me to paste my API key
5. Update the file multiagentz/providers.py with my actual keys
6. Explain what a "stack" YAML config does, then help me create one for my project
   by asking about my code/project directories
7. Test that everything works

Be friendly. Don't assume I know programming. Explain things simply.
If anything fails, tell me what went wrong in plain English and fix it.
```

Claude Code will handle the rest — installing Python, creating the virtual environment,
configuring your API keys, and helping you build your first agent stack.

---

## Step 4: Using multiagentz day to day

Once set up, here's how to use it:

```bash
# 1. Open terminal
# 2. Navigate to your multiagentz folder
cd /path/to/multiagentz-pkg

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Run your stack
maz --config stacks/my-project.yaml
```

Then just type questions and the agent team answers them.

**Useful commands inside maz:**
- `/help` — see all commands
- `/watch /some/folder` — add files for the agents to read
- `/export html` — save the last response as a nice HTML file
- `/brief` — toggle short answers
- `quit` — exit

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `command not found: python3` | Run: `brew install python@3.12` |
| `command not found: brew` | See Step 1c above |
| `command not found: maz` | Make sure you activated the venv: `source .venv/bin/activate` |
| `No API keys configured` | Edit `multiagentz/providers.py` and paste your keys |
| `ModuleNotFoundError` | Re-run: `pip install -e ".[all]"` |
| Anything else | Paste the error into Claude Code — it'll figure it out |

---

## What is this thing, actually?

multiagentz lets you create teams of AI agents that each specialize in different
parts of your codebase. You describe the team in a YAML file:

- **Which folders** each agent knows about
- **What AI model** each agent uses (Claude, Grok, GPT, Gemini)
- **How they're organized** (flat list, nested groups, twin pairs)

When you ask a question, the system automatically figures out which agent(s)
should answer, queries them in parallel, and combines their responses.

Think of it like having a team of specialists instead of one generalist.

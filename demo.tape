# multiagentz (maz) — Quick Start Demo
# Run with: vhs demo.tape
# Requires: vhs (brew install charmbracelet/tap/vhs)

Output demo.gif
Set Theme "Catppuccin Mocha"
Set FontSize 16
Set Width 900
Set Height 600
Set Padding 20
Set TypingSpeed 40ms

# ── Title card ──────────────────────────────────────────────
Type "# multiagentz — Quick Start"
Enter
Sleep 1s

# ── Step 1: Clone ──────────────────────────────────────────
Type "gh repo clone ZlaylowZ/multiagentz"
Enter
Sleep 2s

# ── Step 2: Enter directory ────────────────────────────────
Type "cd multiagentz"
Enter
Sleep 500ms

# ── Step 3: Create venv & activate ─────────────────────────
Type "python3 -m venv .venv"
Enter
Sleep 2s

Type "source .venv/bin/activate"
Enter
Sleep 500ms

# ── Step 4: Install ────────────────────────────────────────
Type 'pip install -e ".[all]"'
Enter
Sleep 4s

# ── Step 5: Setup wizard ──────────────────────────────────
Type "maz setup"
Enter
Sleep 2s

# The wizard prompts for each provider.
# Enter an Anthropic key, skip the rest.
Type "sk-ant-api03-your-key-here"
Enter
Sleep 1s

# Skip OpenAI
Enter
Sleep 500ms

# Skip xAI
Enter
Sleep 500ms

# Skip Google
Enter
Sleep 500ms

# Skip Mistral
Enter
Sleep 2s

# ── Step 6: Run ────────────────────────────────────────────
Type "maz --config stacks/example.yaml"
Enter
Sleep 3s

# Done
Sleep 2s

# demo.tape — VHS script for multiagentz quick-start GIF
# Run with: vhs demo.tape
# Produces: demo.gif

Output demo.gif
Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms

# ── Clone the repo ──────────────────────────────────────────────────────
Type "# Quick Start: multiagentz (maz)"
Enter
Sleep 1s

Type "gh repo clone ZlaylowZ/multiagentz"
Enter
Sleep 2s

Type "cd multiagentz"
Enter
Sleep 500ms

# ── Create venv + install ──────────────────────────────────────────────
Type "python3 -m venv .venv && source .venv/bin/activate"
Enter
Sleep 2s

Type "pip install -e '.[all]'"
Enter
Sleep 4s

# ── Setup wizard ───────────────────────────────────────────────────────
Type "maz setup"
Enter
Sleep 2s

# Enter an Anthropic key, skip the rest
Type "sk-ant-api03-EXAMPLE-KEY-REPLACE-ME"
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

# ── Run maz ────────────────────────────────────────────────────────────
Type "maz --config stacks/example.yaml"
Enter
Sleep 3s

# Ask a question
Type "What does the backend API handle?"
Enter
Sleep 5s

# Exit
Type "/quit"
Enter
Sleep 1s

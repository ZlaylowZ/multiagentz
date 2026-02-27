# multiagentz/main.py
"""
Interactive REPL — the terminal interface for any multi-agent stack.

Usage:
    maz --config stack.yaml
    python -m multiagentz.main --config stack.yaml

Now supports orchestration commands:
    /consensus <question> - Force consensus mode for a query
    /perspective <question> - Execute perspective-based orchestration
    /promote <agent> - Promote agent to LEAD_SUB
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multiagentz.stack import load_stack
from multiagentz.memory import SessionMemory


console = Console()
OUTPUT_DIR = Path("./outputs")
MAX_DISPLAY = 150000


# ── Display helpers ─────────────────────────────────────────────────────

def display_response(response: str, title: str = "[bold blue]Assistant[/bold blue]"):
    """Display response directly to terminal (no animation)."""
    panel = Panel(
        Markdown(response), 
        title=title, 
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)


def export_response(response: str, question: str, fmt: str = "html",
                    title: str = "Assistant") -> Path:
    """Export response to file (completely bypass console output)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in question[:30]).strip("_")
    ext_map = {"html": ".html", "svg": ".svg", "md": ".md", "txt": ".txt"}
    filepath = OUTPUT_DIR / f"{ts}_{slug}{ext_map.get(fmt, '.html')}"

    if fmt == "md":
        filepath.write_text(f"# Query\n\n{question}\n\n---\n\n# Response\n\n{response}")
        return filepath

    # For HTML/SVG/TXT: Use a completely separate Console instance with no terminal access
    from rich.console import Console as OfflineConsole
    import io
    
    # Create a fake file object that NEVER goes to terminal
    null_file = io.StringIO()
    
    # Redirect stderr temporarily to prevent any leaked output
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        export_con = OfflineConsole(
            file=null_file,        # Output to null, not terminal
            record=True,           # But record for export
            width=120,
            stderr=False,          # Don't use stderr
            legacy_windows=False,
            force_terminal=False,
            force_interactive=False,
            force_jupyter=False,
            no_color=False,
            color_system="truecolor" if fmt != "txt" else None,
        )
        
        panel = Panel(
            Markdown(response), 
            title=f"[bold blue]{title}[/bold blue]",
            subtitle=f"[dim]{question[:60]}[/dim]",
            border_style="blue", 
            padding=(1, 2), 
            width=120
        )
        
        # This print goes ONLY to null_file (not terminal)
        export_con.print(panel)
        
        # Export from recorded output
        if fmt == "html":
            html = export_con.export_html(inline_styles=True)
            full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>maz — {question[:80]}</title>
<style>body{{background:#1e1e2e;color:#cdd6f4;font-family:monospace;padding:24px}}</style>
</head><body>{html}</body></html>"""
            filepath.write_text(full)
        elif fmt == "svg":
            filepath.write_text(export_con.export_svg(title="maz"))
        elif fmt == "txt":
            filepath.write_text(export_con.export_text())
    finally:
        # Restore stderr
        sys.stderr = old_stderr

    return filepath


def print_help():

    def print_orchestration_status(lead):
        """Display orchestration configuration and state."""
        table = Table(title="Orchestration Status", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Mode", lead.orchestration_mode)
        table.add_row("Max Iterations", str(lead.orchestration_config.get("max_iterations", 3)))
        
        # LEAD_SUB status
        lead_sub = lead.lead_sub.current_lead_sub
        table.add_row("LEAD_SUB", lead_sub or "(none)")
        
        # Available perspectives
        perspectives = lead.orchestration_config.get("perspectives", [])
        if perspectives:
            p_names = ", ".join(p.get("name", "?") for p in perspectives)
            table.add_row("Configured Perspectives", p_names)
        
        console.print(table)


# ── Main loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Stack REPL")
    parser.add_argument("--config", "-c", required=True, help="Path to stack YAML config")
    args = parser.parse_args()

    lead = load_stack(args.config)
    memory = SessionMemory()
    last_response = None
    last_question = None

    console.print(Panel.fit(
        f"[bold blue]Multi-Agent Stack Ready[/bold blue]\n"
        f"Stack: [cyan]{lead.name}[/cyan]  |  "
        f"Agents: [green]{', '.join(lead.agents.keys())}[/green]\n"
        f"Mode: [yellow]{lead.orchestration_mode}[/yellow]",
        title="🤖 maz",
        border_style="blue",
    ))
    console.print("[dim]Type /help for commands[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()

            if not question:
                continue

            # Exit
            if question.lower() in ("quit", "exit", "q", "end"):
                break

            # Help
            if question.lower() in ("/help", "help", "?"):
                print_help()
                continue

            # Memory
            if question.lower() == "/clear":
                memory.clear()
                console.print("[dim]Memory cleared.[/dim]\n")
                continue

            # Cache
            if question.lower() == "/cache":
                stats = lead._cache.stats()
                console.print(f"[dim]Cache: {stats['entries']} entries, {stats['size_kb']} KB[/dim]\n")
                continue
            if question.lower() == "/cache clear":
                console.print(f"[dim]Cleared {lead._cache.clear()} entries.[/dim]\n")
                continue

            # File input
            if question.lower().startswith("/file "):
                fp = question[6:].strip()
                try:
                    question = Path(fp).expanduser().read_text().strip()
                    console.print(f"[dim]Loaded {len(question)} chars[/dim]\n")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]\n")
                    continue

            # Watch commands (files agent)
            files_agent = lead.agents.get("files")
            if question.lower().startswith("/watch ") and files_agent:
                console.print(f"[dim]{files_agent.add_path(question[7:].strip())}[/dim]\n")
                continue
            if question.lower().startswith("/unwatch ") and files_agent:
                console.print(f"[dim]{files_agent.remove_path(question[9:].strip())}[/dim]\n")
                continue
            if question.lower() == "/watched" and files_agent:
                paths = files_agent.list_watched()
                for p in (paths or ["(none)"]):
                    console.print(f"[dim]  {p}[/dim]")
                console.print()
                continue
            if question.lower() == "/clear-watched" and files_agent:
                console.print(f"[dim]{files_agent.clear_watched()}[/dim]\n")
                continue
            if question.lower() == "/context" and files_agent:
                stats = files_agent.get_context_stats()
                console.print(f"[dim]{stats['context_chars']:,} chars (~{stats['context_tokens_approx']:,} tokens)[/dim]\n")
                continue
            if question.lower().startswith("/scan ") and files_agent:
                path = question[6:].strip()
                console.print(f"[dim]{files_agent.add_path(path)}[/dim]")
                console.print("[dim]Generating summary...[/dim]\n")
                summary = files_agent.query(
                    f"Provide a concise overview of {path}. "
                    "What is this project? Tech stack? Main components?"
                )
                display_response(summary, title="[bold green]Scan Results[/bold green]")
                console.print()
                continue

            # Paste mode
            if question.lower() == "/paste":
                console.print("[dim]Paste content, then type END on its own line.[/dim]\n")
                lines = []
                while True:
                    line = sys.stdin.readline()
                    if line == "" or line.strip().upper() == "END":
                        break
                    lines.append(line.rstrip("\r\n"))
                question = "\n".join(lines)
                console.print(f"[dim]Captured {len(question):,} chars[/dim]\n")

            # Brief mode
            if question.lower() == "/brief":
                lead.brief_mode = not lead.brief_mode
                console.print(f"[dim]Brief mode: {'ON' if lead.brief_mode else 'OFF'}[/dim]\n")
                continue

            # Export
            if question.lower().startswith("/export"):
                if not last_response:
                    console.print("[dim]No response to export yet.[/dim]\n")
                    continue
                parts = question.split()
                fmt = parts[1].lower() if len(parts) > 1 else "html"
                fp = export_response(last_response, last_question, fmt=fmt)
                console.print(f"[bold green]Exported:[/bold green] {fp}\n")
                continue

            # ── ORCHESTRATION COMMANDS ──

            # Status
            if question.lower() == "/status":
                print_orchestration_status(lead)
                console.print()
                continue

            # Promote to LEAD_SUB
            if question.lower().startswith("/promote "):
                agent_name = question[9:].strip()
                result = lead.lead_sub.promote(agent_name)
                console.print(f"[cyan]{result}[/cyan]\n")
                continue

            # Demote LEAD_SUB
            if question.lower() == "/demote":
                result = lead.lead_sub.demote()
                console.print(f"[cyan]{result}[/cyan]\n")
                continue

            # Consensus mode
            if question.lower().startswith("/consensus "):
                actual_question = question[11:].strip()
                if not actual_question:
                    console.print("[red]Usage: /consensus <question>[/red]\n")
                    continue
                
                console.print("[cyan]Executing consensus synthesis mode...[/cyan]\n")
                memory.add_user(actual_question)
                
                response, metadata = lead.orchestration.execute_consensus(
                    actual_question,
                    memory=memory,
                    max_iterations=lead.orchestration_config.get("max_iterations", 3)
                )
                
                agents_used = metadata.get("agents_used", [])
                memory.add_assistant(response, routed_to=agents_used)
                
                # Display metadata
                console.print(f"[dim]Iterations: {metadata.get('iterations', 0)} | "
                              f"Conflicts: {len(metadata.get('conflicts_found', []))} | "
                              f"Consensus: {metadata.get('consensus_achieved', False)}[/dim]\n")
                
                last_response = response
                last_question = actual_question
                
                # Display response to terminal
                if len(response) > MAX_DISPLAY:
                    display_response(response[:2000] + "\n\n... [truncated] ...",
                                     title="[bold blue]Consensus Result (Preview)[/bold blue]")
                else:
                    display_response(response, title="[bold blue]Consensus Result[/bold blue]")
                
                # Export to HTML
                filepath = export_response(response, actual_question, fmt="html")
                console.print(f"\n[dim]Saved: {filepath}[/dim]\n")
                continue

            # Perspective mode
            if question.lower().startswith("/perspective "):
                parts = question[13:].strip().split('"')
                if len(parts) < 3:
                    console.print("[red]Usage: /perspective \"<question>\" [agent names...][/red]")
                    console.print("[dim]Example: /perspective \"Design auth flow\" sub_mem sub_inc[/dim]\n")
                    continue
                
                actual_question = parts[1]
                agent_names = parts[2].strip().split() if len(parts) > 2 else []
                
                # Get perspective configs from stack or use specified agents
                if not agent_names:
                    # Use configured perspectives from YAML
                    perspective_configs = lead.orchestration_config.get("perspectives", [])
                    if not perspective_configs:
                        console.print("[red]No perspectives configured in stack YAML[/red]\n")
                        continue
                else:
                    # Build perspective configs from agent names
                    perspective_configs = []
                    for i, agent_name in enumerate(agent_names):
                        if agent_name not in lead.agents:
                            console.print(f"[red]Agent '{agent_name}' not found[/red]\n")
                            continue
                        perspective_configs.append({
                            "name": f"perspective_{i+1}_{agent_name}",
                            "agent_ref": agent_name,
                            "memory_access": "shared" if i == 0 else "none",
                            "role": f"Perspective {i+1}"
                        })
                
                if not perspective_configs:
                    console.print("[red]No valid perspectives to execute[/red]\n")
                    continue
                
                console.print(f"[cyan]Executing perspective mode with {len(perspective_configs)} perspectives...[/cyan]\n")
                memory.add_user(actual_question)
                
                response, metadata = lead.query_perspective(
                    actual_question,
                    perspective_configs,
                    memory=memory,
                    bootstrap_qa=True
                )
                
                memory.add_assistant(response, routed_to=metadata.get("perspectives", []))
                
                # Display metadata
                console.print(f"[dim]Perspectives: {', '.join(metadata.get('perspectives', []))} | "
                              f"Iterations: {metadata.get('convergence_iterations', 0)} | "
                              f"Converged: {metadata.get('converged', False)}[/dim]\n")
                
                last_response = response
                last_question = actual_question
                
                # Display response to terminal
                if len(response) > MAX_DISPLAY:
                    display_response(response[:2000] + "\n\n... [truncated] ...",
                                     title="[bold blue]Perspective Result (Preview)[/bold blue]")
                else:
                    display_response(response, title="[bold blue]Perspective Result[/bold blue]")
                
                # Export to HTML
                filepath = export_response(response, actual_question, fmt="html")
                console.print(f"\n[dim]Saved: {filepath}[/dim]\n")
                continue

            # Unknown commands
            if question.startswith("/"):
                console.print(f"[red]Unknown command: {question.split()[0]}[/red]")
                console.print("[dim]Type /help for commands.[/dim]\n")
                continue

            # ── Process normal question ──
            memory.add_user(question)
            response, agents_used = lead.query(question, memory=memory)
            memory.add_assistant(response, routed_to=agents_used)

            last_response = response
            last_question = question

            # Display response to terminal
            console.print()  # Blank line before response
            if len(response) > MAX_DISPLAY:
                display_response(response[:2000] + "\n\n... [truncated] ...",
                                 title="[bold blue]Assistant (Preview)[/bold blue]")
            else:
                display_response(response)

            # Export to HTML
            filepath = export_response(response, question, fmt="html")
            console.print(f"\n[dim]Saved: {filepath}[/dim]")
            console.print("[dim]/export [fmt] to re-export[/dim]\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]\n")

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
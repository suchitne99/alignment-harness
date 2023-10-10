import typer
from rich.console import Console
from rich.table import Table
from alignment_harness.evaluators.elo import EloRatingEngine

app = typer.Typer(help="Alignment Harness CLI")
console = Console()

@app.command()
def leaderboard():
    engine = EloRatingEngine()
    engine.record_match("llama-3-70b-instruct", "mistral-7b-instruct", "model_a")
    engine.record_match("qwen-2.5-72b", "mistral-7b-instruct", "model_a")
    table = Table(title="LLM-as-a-Judge ELO Leaderboard")
    table.add_column("Model", style="cyan")
    table.add_column("ELO", style="green")
    for name, item in engine.ratings.items():
        table.add_row(name, f"{item.elo:.1f}")
    console.print(table)

if __name__ == "__main__":
    app()

# Column width optimization

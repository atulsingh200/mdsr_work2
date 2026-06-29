"""CLI for inspecting and downloading datasets.

Examples:
    followup-data list
    followup-data info qrecc
    followup-data download clariq qrecc
    followup-data head qrecc -n 3
    followup-data export qrecc --split train -o data/qrecc_train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .base import DatasetNotDownloaded
from .registry import default_root, get, list_datasets, load

app = typer.Typer(help="Follow-up causal embedding dataset CLI")
console = Console()


def _resolve_root(root: str | None) -> Path:
    return Path(root).expanduser().resolve() if root else default_root()


def _load_with_fallback(name: str, root: Path, split: str):
    """Load with the requested split, or fall back to the first declared split."""
    cls = get(name)
    chosen = split if split in cls.splits else cls.splits[0]
    return load(name, root=root, split=chosen)


@app.command("list")
def cmd_list() -> None:
    """List registered datasets."""
    table = Table(title="Registered datasets", show_lines=False)
    table.add_column("name", style="bold cyan")
    table.add_column("homepage")
    table.add_column("description")
    for n in list_datasets():
        cls = get(n)
        table.add_row(n, cls.homepage, cls.description.replace("\n", " ").strip())
    console.print(table)


@app.command("info")
def cmd_info(name: str) -> None:
    """Show metadata for one dataset."""
    cls = get(name)
    console.print(f"[bold cyan]{cls.name}[/]")
    console.print(f"  homepage : {cls.homepage}")
    console.print(f"  citation : {cls.citation}")
    console.print(f"  license  : {cls.license}")
    console.print(f"  splits   : {cls.splits}")
    console.print(f"  desc     : {cls.description}")


@app.command("download")
def cmd_download(
    names: list[str] = typer.Argument(..., help="One or more dataset names"),
    root: str = typer.Option(None, "--root", help="Data root (default: ./data)"),
    split: str = typer.Option("train", "--split"),
) -> None:
    """Download one or more datasets."""
    rootp = _resolve_root(root)
    for n in names:
        ds = _load_with_fallback(n, rootp, split)
        console.print(f"[bold]downloading[/] {n} into {ds.data_dir}")
        try:
            ds.download()
            console.print(f"  [green]ok[/] is_downloaded={ds.is_downloaded()}")
        except Exception as e:
            console.print(f"  [red]failed[/] {type(e).__name__}: {e}")


@app.command("head")
def cmd_head(
    name: str,
    n: int = typer.Option(3, "-n", "--n"),
    root: str = typer.Option(None, "--root"),
    split: str = typer.Option("train", "--split"),
) -> None:
    """Print the first n examples."""
    rootp = _resolve_root(root)
    ds = _load_with_fallback(name, rootp, split)
    try:
        examples = ds.head(n)
    except DatasetNotDownloaded as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    for i, ex in enumerate(examples):
        console.rule(f"{name} #{i}")
        console.print(f"[bold]anchor   [/] {ex.anchor}")
        console.print(f"[bold]positive [/] {ex.positive}")
        if ex.context:
            console.print(f"[dim]context ({len(ex.context)} prior turns)[/]")
        if ex.metadata:
            console.print(f"[dim]meta {ex.metadata}[/]")


@app.command("export")
def cmd_export(
    name: str,
    out: Path = typer.Option(..., "-o", "--out"),
    root: str = typer.Option(None, "--root"),
    split: str = typer.Option("train", "--split"),
) -> None:
    """Export a dataset to JSONL of PairExamples."""
    rootp = _resolve_root(root)
    ds = _load_with_fallback(name, rootp, split)
    n = ds.to_jsonl(out)
    console.print(f"wrote {n} pairs to {out}")


@app.command("triples")
def cmd_triples(
    name: str,
    k: int = typer.Option(4, "-k", help="negatives per anchor"),
    n: int = typer.Option(3, "-n", help="how many to print"),
    root: str = typer.Option(None, "--root"),
    split: str = typer.Option("train", "--split"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Print sample (anchor, positive, negatives[k]) triples."""
    from .negatives import with_random_negatives

    rootp = _resolve_root(root)
    ds = _load_with_fallback(name, rootp, split)
    pairs = list(ds.head(max(n, 64)))  # small pool to keep CLI fast
    triples = list(with_random_negatives(pairs, k=k, seed=seed))[:n]
    for i, t in enumerate(triples):
        console.rule(f"{name} triple #{i}")
        console.print(f"[bold green]anchor   [/] {t.anchor}")
        console.print(f"[bold green]positive [/] {t.positive}")
        for j, neg in enumerate(t.negatives):
            console.print(f"[red]neg {j}    [/] {neg}")


if __name__ == "__main__":
    app()

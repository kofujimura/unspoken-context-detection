#!/usr/bin/env python3
"""
CLI for UCC detection system.
Based on SPEC.md Section 9.1.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from backend.core.models import DetectionParams
from backend.embedding import OpenAIEmbedder
from backend.batch import UCCPipeline

# Load environment
load_dotenv()

# Initialize CLI
app = typer.Typer(
    name="ucc-detect",
    help="Unspoken Context Detection CLI",
    add_completion=False
)
console = Console()


@app.command()
def analyze(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to subtitle file (.vtt or .srt)",
        exists=True
    ),
    video_id: str = typer.Option(
        ...,
        "--video-id",
        help="Video identifier"
    ),
    language: str = typer.Option(
        "ja",
        "--language",
        "-l",
        help="Language code (ja, en, etc.)"
    ),
    output_dir: Path = typer.Option(
        "output",
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    k: int = typer.Option(
        5,
        "--k",
        help="Window size"
    ),
    lambda_param: float = typer.Option(
        1.0,
        "--lambda",
        help="Threshold multiplier"
    ),
    tau_topic: Optional[float] = typer.Option(
        0.55,
        "--tau-topic",
        help="Topic continuity threshold (None to disable)"
    ),
    mark_span_start: int = typer.Option(
        0,
        "--mark-span-start",
        help="Marking span start offset"
    ),
    mark_span_end: int = typer.Option(
        2,
        "--mark-span-end",
        help="Marking span end offset"
    ),
    n_reasons: int = typer.Option(
        5,
        "--n-reasons",
        help="Number of reason keywords to extract"
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable embedding cache"
    ),
    export_html: bool = typer.Option(
        True,
        "--export-html/--no-html",
        help="Export HTML viewer"
    )
):
    """
    Analyze subtitle file for UCC.

    Example:
        ucc-detect analyze --input subtitles.vtt --video-id lecture_01 --language ja
    """
    # Validate input
    if not input_file.suffix.lower() in ['.vtt', '.srt']:
        console.print(f"[red]Error: Unsupported file format {input_file.suffix}[/red]")
        raise typer.Exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create parameters
    params = DetectionParams(
        k=k,
        lambda_param=lambda_param,
        tau_topic=tau_topic,
        mark_span=[mark_span_start, mark_span_end],
        n_reasons=n_reasons
    )

    # Display configuration
    console.print("\n[bold cyan]Configuration:[/bold cyan]")
    config_table = Table(show_header=False)
    config_table.add_row("Input File", str(input_file))
    config_table.add_row("Video ID", video_id)
    config_table.add_row("Language", language)
    config_table.add_row("Window Size (k)", str(k))
    config_table.add_row("Lambda (λ)", str(lambda_param))
    config_table.add_row("Tau Topic", str(tau_topic) if tau_topic is not None else "disabled")
    config_table.add_row("Output Directory", str(output_dir))
    console.print(config_table)

    # Run analysis
    console.print("\n[bold cyan]Running analysis...[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        # Run async pipeline
        result = asyncio.run(_run_analysis(
            input_file,
            video_id,
            language,
            params,
            use_cache=not no_cache
        ))

        progress.update(task, completed=True)

    if not result.success:
        console.print(f"\n[red]Error: {result.error}[/red]")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Analysis Complete![/bold green]\n")

    stats_table = Table(title="Results Summary")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")

    metadata = result.result.metadata
    stats_table.add_row("Total Segments", str(metadata['total_segments']))
    stats_table.add_row("Marked Segments", str(metadata['marked_count']))
    stats_table.add_row(
        "Marked %",
        f"{metadata['marked_count'] / metadata['total_segments'] * 100:.1f}%"
    )
    stats_table.add_row("Threshold (ΔD)", f"{metadata['threshold']:.3f}")
    stats_table.add_row("Mean (μ)", f"{metadata['mu']:.3f}")
    stats_table.add_row("Std Dev (σ)", f"{metadata['sigma']:.3f}")

    console.print(stats_table)

    # Export results
    json_path = output_dir / f"{video_id}_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result.result.model_dump(), f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]✓[/green] JSON result saved to: {json_path}")

    # Export HTML if requested
    if export_html:
        html_path = output_dir / f"{video_id}_viewer.html"
        _export_html_viewer(result.result, html_path)
        console.print(f"[green]✓[/green] HTML viewer saved to: {html_path}")

    # Show warnings
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")

    # Show sample marked segments
    marked = result.result.marked_segments[:5]
    if marked:
        console.print(f"\n[bold cyan]Sample Marked Segments (showing {len(marked)}):[/bold cyan]\n")
        for seg in marked:
            console.print(f"[yellow]#{seg.id}[/yellow] [{_format_time(seg.start_ms)} - {_format_time(seg.end_ms)}]")
            console.print(f"  {seg.text}")
            if seg.reasons:
                console.print(f"  [dim]Missing: {', '.join(seg.reasons)}[/dim]")
            console.print()


async def _run_analysis(
    input_file: Path,
    video_id: str,
    language: str,
    params: DetectionParams,
    use_cache: bool = True
):
    """Run async analysis pipeline."""
    embedder = OpenAIEmbedder()
    pipeline = UCCPipeline(embedder, params)

    return await pipeline.analyze_file(
        str(input_file),
        video_id,
        language
    )


def _export_html_viewer(result, output_path: Path):
    """Export standalone HTML viewer with embedded data."""
    # Read viewer template
    template_path = Path(__file__).parent / "frontend" / "viewer.html"

    if not template_path.exists():
        console.print("[yellow]Warning: viewer.html template not found, skipping HTML export[/yellow]")
        return

    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Read viewer.js
    js_path = Path(__file__).parent / "frontend" / "viewer.js"
    with open(js_path, 'r', encoding='utf-8') as f:
        js = f.read()

    # Inject data and make standalone
    result_json = json.dumps(result.model_dump(), ensure_ascii=False)

    standalone_html = html.replace(
        '<script src="viewer.js"></script>',
        f'''<script>
// Embedded analysis result
const embeddedResult = {result_json};

// Auto-load embedded result
window.addEventListener('DOMContentLoaded', () => {{
    if (embeddedResult) {{
        analysisResult = embeddedResult;
        displayResults();
        document.querySelector('.upload-section').style.display = 'none';
    }}
}});

{js}
</script>'''
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(standalone_html)


def _format_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS."""
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60

    s = seconds % 60
    m = minutes % 60

    if hours > 0:
        return f"{hours}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"


@app.command()
def version():
    """Show version information."""
    console.print("[bold cyan]UCC Detection System[/bold cyan]")
    console.print("Version: 1.0.0")
    console.print("Based on: Unspoken Context Change Detection Framework")


if __name__ == "__main__":
    app()

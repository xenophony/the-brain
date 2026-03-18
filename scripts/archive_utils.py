"""Shared archive utility for results directories."""

from datetime import datetime
from pathlib import Path


def archive_previous_run(output_dir: Path, file_glob: str = "*.json") -> Path | None:
    """Move previous run's files into archive/<timestamp>/.

    Archives all files matching file_glob plus any plots/ subdirectory.
    Returns the archive directory path, or None if nothing to archive.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    # Check if there are files to archive (skip archive/ and other subdirs)
    files_to_move = [f for f in output_dir.iterdir()
                     if f.is_file() and f.name != ".gitkeep"]
    plots_dir = output_dir / "plots"
    has_plots = plots_dir.exists() and any(plots_dir.iterdir())

    if not files_to_move and not has_plots:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in files_to_move:
        f.rename(archive_dir / f.name)
        moved += 1

    if has_plots:
        archive_plots = archive_dir / "plots"
        plots_dir.rename(archive_plots)
        moved += 1

    if moved:
        print(f"Archived {moved} items from previous run → {archive_dir}")

    return archive_dir

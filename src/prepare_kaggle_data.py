import os
import zipfile
import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()

def prepare_plastic_dataset():
    """
    Downloads Kaggle dataset, extracts only plastic bottle images,
    and saves them into data/input_data/ as image_{i}.jpg
    """
    home = Path.home()
    zip_path = home / "Downloads" / "glass-and-plastic-bottles.zip"
    extract_dir = Path("data/raw_dataset")
    output_dir = Path("data/input_data")

    extract_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download dataset if not exists
    if not zip_path.exists():
        console.log("[yellow]Downloading dataset from Kaggle...[/yellow]")
        subprocess.run([
            "curl", "-L", "-o", str(zip_path),
            "https://www.kaggle.com/api/v1/datasets/download/antonpivnenko/glass-and-plastic-bottles"
        ], check=True)
    else:
        console.log(f"[green]Dataset already exists at {zip_path}[/green]")

    # Step 2: Extract dataset
    console.log("[yellow]Extracting dataset...[/yellow]")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Step 3: Locate the correct plastic dataset folder
    plastic_dir = None
    for root, dirs, files in os.walk(extract_dir):
        # Only consider folders explicitly named "plastic"
        if os.path.basename(root).lower() == "plastic":
            plastic_dir = Path(root)
            break

    if not plastic_dir:
        console.log("[red]Plastic dataset not found![/red]")
        return []

    console.log(f"[green]Using plastic dataset at: {plastic_dir}[/green]")

    # Step 4: Collect and rename images
    images = [f for f in plastic_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not images:
        console.log(f"[red]No images found in {plastic_dir}[/red]")
        return []

    console.log(f"[blue]Found {len(images)} plastic images[/blue]")

    saved_paths = []
    for idx, img_path in enumerate(track(images, description="Copying images...")):
        new_name = f"image_{idx+1}.jpg"
        new_path = output_dir / new_name
        shutil.copy(img_path, new_path)
        saved_paths.append(new_path)

    console.log(f"[bold green]All plastic images saved to {output_dir}[/bold green]")
    return saved_paths


# Example usage
if __name__ == "__main__":
    files = prepare_plastic_dataset()
    console.log(f"[cyan]Sample files:[/cyan] {files[:5]}")

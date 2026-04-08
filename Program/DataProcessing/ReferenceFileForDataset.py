import cv2
import base64
from pathlib import Path
from collections import defaultdict

"""
This script generates an HTML file that allows visual reference to understand the dataset.
It scans the dataset folder, extracts one frame from each video, and organizes them by label.
"""

# ------------- Path Configuration -------------

DATASET_DIR = r"Program\Datasets\BalancedDataset" 
OUTPUT_DIR  = r"Program\Datasets\FrameReference"   

# --------- Additional Configurations ----------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
FRAME_POSITION   = 0.3   
THUMB_WIDTH      = 180   # Thumbnail width

# -------------- Frame Processing --------------

def get_all_videos(dataset_dir: Path) -> dict:
    """Walk dataset_dir and group video paths by label (parent folder name)."""
    groups = defaultdict(list)
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            label = path.parent.name   
            groups[label].append(path)
    return dict(sorted(groups.items()))


def extract_frame(video_path: Path):
    """Grab one frame at FRAME_POSITION. Returns JPEG bytes or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open: {video_path.name}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * FRAME_POSITION) - 1))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"  [WARN] Could not read frame: {video_path.name}")
        return None

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def to_data_uri(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


# ------------- HTML Creator -------------

def build_html(groups: dict) -> str:
    total_videos = sum(len(v) for v in groups.values())
    total_labels = len(groups)

    sections = []
    for label, videos in groups.items():
        cards = ""
        count = 0
        for vpath in videos:
            jpeg = extract_frame(vpath)
            if jpeg is None:
                continue
            cards += f"""
            <div class="card">
              <img src="{to_data_uri(jpeg)}" alt="{label}">
              <div class="caption">{vpath.stem[:20]}</div>
            </div>"""
            count += 1
            print(f"  ok  [{label}]  {vpath.name}")

        sections.append(f"""
        <section id="{label}">
          <h2>{label} <span class="count">({count} videos)</span></h2>
          <div class="grid">{cards}</div>
        </section>""")

    toc = "".join(f'<li><a href="#{l}">{l}</a></li>' for l in groups)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Dataset Frame Reference</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              background: #f4f4f6; color: #222; padding: 2rem; }}
    header {{ background: #1a1a2e; color: #fff; padding: 1.5rem 2rem;
              border-radius: 10px; margin-bottom: 2rem; }}
    header h1 {{ font-size: 1.6rem; margin-bottom: .4rem; }}
    header p  {{ font-size: .9rem; opacity: .75; }}
    .stats {{ display: flex; gap: 2rem; margin-top: 1rem; }}
    .stat  {{ background: rgba(255,255,255,.1); padding: .5rem 1rem;
              border-radius: 6px; font-size: .85rem; }}
    .stat b {{ font-size: 1.1rem; display: block; }}
    nav  {{ background: #fff; border-radius: 8px; padding: 1rem 1.5rem;
             margin-bottom: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    nav h3 {{ margin-bottom: .6rem; font-size: .9rem; color: #555; }}
    nav ul {{ display: flex; flex-wrap: wrap; gap: .4rem; list-style: none; }}
    nav a  {{ text-decoration: none; background: #eef; color: #33a;
              padding: .25rem .7rem; border-radius: 4px; font-size: .85rem; }}
    nav a:hover {{ background: #dde; }}
    section {{ background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem;
               margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    section h2 {{ font-size: 1.3rem; border-bottom: 2px solid #eee;
                  padding-bottom: .4rem; margin-bottom: 1rem; }}
    .count {{ font-size: .85rem; color: #888; font-weight: normal; }}
    .grid  {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .card  {{ display: flex; flex-direction: column; align-items: center; }}
    .card img {{ border-radius: 5px; object-fit: cover; border: 1px solid #ddd;
                 height: {int(THUMB_WIDTH * 0.75)}px; width: {THUMB_WIDTH}px; }}
    .caption {{ font-size: .7rem; color: #666; margin-top: 3px; text-align: center;
                width: {THUMB_WIDTH}px; overflow: hidden; text-overflow: ellipsis;
                white-space: nowrap; }}
  </style>
</head>
<body>
  <header>
    <h1>Dataset Frame Reference</h1>
    <p>One frame per video &middot; {total_labels} labels &middot; {total_videos} videos</p>
    <div class="stats">
      <div class="stat"><b>{total_labels}</b>Labels</div>
      <div class="stat"><b>{total_videos}</b>Videos</div>
      <div class="stat"><b>{int(FRAME_POSITION*100)}%</b>Frame position</div>
    </div>
  </header>
  <nav>
    <h3>Jump to label</h3>
    <ul>{toc}</ul>
  </nav>
  {"".join(sections)}
</body>
</html>"""


# ------------- Output -------------

def main():
    dataset_dir = Path(DATASET_DIR).resolve()
    output_dir  = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print("        Double-check DATASET_DIR at the top of the file.")
        return

    print(f"\nScanning: {dataset_dir}")
    groups = get_all_videos(dataset_dir)

    if not groups:
        print("[ERROR] No videos found. Check VIDEO_EXTENSIONS and your folder structure.")
        return

    total = sum(len(v) for v in groups.values())
    print(f"   Found {len(groups)} labels, {total} videos\n")

    html     = build_html(groups)
    out_path = output_dir / "dataset_reference.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\nDone!  {out_path}  ({size_mb:.1f} MB)")
    print("Open dataset_reference.html in any browser.\n")


if __name__ == "__main__":
    main()
import csv
from pathlib import Path

MANIFEST_PATH = Path("data/input_manifest.csv")


def update_manifest():
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        clip_id = row["clip_id"]
        row["file_name"] = f"data/raw/{clip_id}.mp4"

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Successfully updated {len(rows)} entries in {MANIFEST_PATH}")


if __name__ == "__main__":
    update_manifest()

from urllib.request import urlretrieve, Request, urlopen
from pathlib import Path
import time
import zipfile
import shutil
import numpy as np


def get_paths():
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw" 
    zip_path = raw_dir / "ncaltech101" / "cy6cvx3ryv-1.zip"
    return repo, raw_dir, zip_path

def download(url: str, out_path: Path, chunk_bytes: int = 8 * 1024 * 1024):
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if out_path.exists() and out_path.stat().st_size > 0:
        print("[skip] ", out_path, " already exists.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    start = time.time()
    downloaded = 0
    with urlopen(req) as r:
        total = r.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        with open(tmp_path, "wb") as f:
            while True:
                chunk = r.read(chunk_bytes)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    pct = 100 * downloaded/total_bytes
                    print(f"\r{pct:6.2f}% ({downloaded}/{total_bytes} bytes)", end="")
                else:
                    print(f"\r{downloaded} bytes", end= "")

            print()
            tmp_path.replace(out_path)
            elapsed = time.time() - start
            print(f"[downloaded] {out_path} in {elapsed:.1f}s.")

def unzip(zip_path):
    assert zip_path.exists()
    dest_dir = zip_path.parent
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Unzipped file.")
    zip_path.unlink()
    print("Cleaned up ", zip_path.name)

    # Check for wrapper
    items = list(dest_dir.iterdir())
    wrapper_path = dest_dir / "cy6cvx3ryv-1"
    if wrapper_path.exists() and wrapper_path.is_dir(): 
        print("flattening wrapper folder...")
        for file_path in wrapper_path.iterdir():
            shutil.move(str(file_path), str(dest_dir))
        wrapper_path.rmdir()
        print("Folder flattened.")

    # Unzip nested folders
    events_zip = dest_dir / "Caltech101.zip"
    annotations_zip = dest_dir / "Caltech101_annotations.zip"
    events = dest_dir / "events" 
    annotations = dest_dir / "annotations"
    with zipfile.ZipFile(events_zip, "r") as zf:
        zf.extractall(events)
        print("Unzipped file.")
    events_zip.unlink()
    print("Cleaned up ", events_zip.name)
    with zipfile.ZipFile(annotations_zip, "r") as zf:
        zf.extractall(annotations)
        print("Unzipped file.")
    annotations_zip.unlink()
    print("Cleaned up ", annotations_zip.name)


    # Check for wrapper
    items = list(events.iterdir())
    wrapper_path = events / "Caltech101"
    if wrapper_path.exists() and wrapper_path.is_dir(): 
        print("flattening wrapper folder...")
        for file_path in wrapper_path.iterdir():
            shutil.move(str(file_path), str(events))
        wrapper_path.rmdir()
        print("Folder flattened.")
        
    # Check for wrapper
    items = list(annotations.iterdir())
    wrapper_path = annotations / "Caltech101_annotations"
    if wrapper_path.exists() and wrapper_path.is_dir(): 
        print("flattening wrapper folder...")
        for file_path in wrapper_path.iterdir():
            shutil.move(str(file_path), str(annotations))
        wrapper_path.rmdir()
        print("Folder flattened.")

    # Keep only two classes
    filter_dataset(events)
    filter_dataset(annotations)


def filter_dataset(data_dir: Path):
    KEEP = {"Faces_easy", "Motorbikes"}
    print("filtering dataset... Keeping only: ", KEEP)
    for item in data_dir.iterdir():
        if item.is_dir():
            if item.name not in KEEP:
                shutil.rmtree(item)
            else:
                print("[Kept]",item.name)

def decode_dataset(raw: Path):
    decoded = raw.parent / "decoded_npz"
    events = raw / "ncaltech101" / "events"

    decoded.mkdir(parents=True, exist_ok=True)

    bin_files = list(events.rglob("*.bin"))
    total = len(bin_files)
    print(len(bin_files)," .bin files found")
    for i, bf in enumerate(bin_files):
        print(f"\r{i} out of {total} .bin files deocded.",end="")
        rel = bf.relative_to(raw)
        out_path = (decoded / rel).with_suffix(".npz")
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        x, y, p, t = decode_bin(bf)

#        if i<5:
#            print("num events:", len(t))
#            print("x range:", int(x.min()), "→", int(x.max()))
#            print("y range:", int(y.min()), "→", int(y.max()))
#
#            print("polarity values:", set(p.tolist()))
#
#            print("t range (µs):", int(t.min()), "→", int(t.max()))

        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, x=x, y=y, p=p, t=t)
        tmp.replace(out_path)

        z = np.load(out_path)
        x1, y1, p1, t1 = z["x"], z["y"], z["p"], z["t"]
        assert x1.shape == x.shape
        assert (p1 == p).all()

        

def decode_bin(bin_file):
   data = bin_file.read_bytes()
   if len(data) % 5 != 0:
       raise Valueerror(f"{bin_file} size {len(data)} not divisible by 5.")
   arr = np.frombuffer(data, dtype=np.uint8).reshape(-1,5)
   b0 = arr[:,0]
   b1 = arr[:,1]
   b2 = arr[:,2]
   b3 = arr[:,3]
   b4 = arr[:,4]

   x=b0.astype(np.uint16)
   y=b1.astype(np.uint16)

   p = (((b2 & 0x80) >> 7).astype(np.uint8))

   t = (b2 & 0x7f).astype(np.uint32) << 16 | (b3.astype(np.uint32)) << 8 | (b4.astype(np.uint32))

   # Normalize timestaps
   t = t - t.min()

   return x, y, p, t
    
def main():
     repo, raw_dir, zip_path = get_paths()
     raw_dir.mkdir(parents=True, exist_ok=True)
     URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cy6cvx3ryv-1.zip"
     check_events = raw_dir / "ncaltech101" / "events"
     check_annotations = raw_dir / "ncaltech101" / "annotations"
     if not(check_events.exists() and check_annotations.exists() and any(check_events.iterdir()) and any(check_annotations.iterdir())):
         download(URL, zip_path)
         unzip(zip_path)
     data_path = zip_path.parent
     decode_dataset(raw_dir)
     

if __name__ == "__main__":
    main()

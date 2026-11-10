"""Fetch one official nnU-Net fold and one public MSD CT for a reproducible demo."""
import io
import hashlib
import json
import tarfile
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://zenodo.org/records/4003545/files/Task009_Spleen.zip?download=1"
DATA_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
PREFIX = "3d_fullres/Task009_Spleen/nnUNetTrainerV2__nnUNetPlansv2.1/"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        self.session = requests.Session()
        self.cache_start, self.cache = 0, b""
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        if whence == 0: self.position = offset
        elif whence == 1: self.position += offset
        else: raise ValueError("SEEK_END unsupported")
        return self.position
    def read(self, size=-1):
        if size < 0: raise ValueError("bounded reads required")
        if not (self.cache_start <= self.position and self.position + size <= self.cache_start + len(self.cache)):
            end = self.position + max(size, 65536) - 1
            response = self.session.get(self.url, headers={"Range": f"bytes={self.position}-{end}"}, timeout=120)
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {self.position}-"):
                raise RuntimeError("Server did not honor HTTP Range")
            self.cache_start, self.cache = self.position, response.content
        start = self.position - self.cache_start
        data = self.cache[start:start+size]
        self.position += len(data)
        return data


def main():
    target = ROOT / "models" / "nnunet"
    target.mkdir(parents=True, exist_ok=True)
    files = [PREFIX + "plans.pkl", PREFIX + "fold_0/model_final_checkpoint.model.pkl", PREFIX + "fold_0/model_final_checkpoint.model"]
    with RemoteZip(MODEL_URL, timeout=120) as archive:
        records = []
        for name in files:
            info = archive.getinfo(name)
            path = target / name
            if not path.exists() or path.stat().st_size != info.file_size:
                print(f"Downloading {name}: {info.file_size} bytes", flush=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(name)  # ZipFile verifies member CRC32.
                temp = path.with_suffix(path.suffix + ".part")
                temp.write_bytes(data)
                temp.replace(path)
            records.append({"file": name, "size": info.file_size, "crc32": info.CRC,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"source": MODEL_URL, "archive_md5": "3a7bef6d372b6fd9e1d7454a23bd793f",
                "note": "Only selected members downloaded; member CRC verified, full archive MD5 not checked.", "files": records}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sample_dir = ROOT / "data" / "demo"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_dir / "ct.nii.gz"
    if not sample.exists():
        with RangeReader(DATA_URL) as raw:
            with tarfile.open(fileobj=raw, mode="r:") as archive:
                for member in archive:
                    if "/imagesTr/" in member.name and member.name.endswith(".nii.gz") and "/._" not in member.name:
                        print(f"Downloading public demonstration CT: {member.name}", flush=True)
                        temp = sample.with_suffix(".part")
                        temp.write_bytes(archive.extractfile(member).read())
                        temp.replace(sample)
                        (sample_dir / "source.json").write_text(json.dumps({"url": DATA_URL, "member": member.name,
                            "note": "Public MSD training sample, for pipeline demonstration only; not a held-out accuracy evaluation."}, indent=2))
                        break
    if not sample.exists():
        raise RuntimeError("No CT found in sample dataset")
    print("Demo model and CT ready", flush=True)


if __name__ == "__main__":
    main()

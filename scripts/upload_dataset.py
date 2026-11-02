"""Upload a local JSONL as a Fireworks dataset: create the entry, upload through a
signed URL, validate, then wait for READY.

    .venv-fw/bin/python scripts/upload_dataset.py <dataset-id> <path.jsonl>
"""
import os, sys, time
import httpx
from dotenv import dotenv_values
from fireworks import Fireworks

ACCOUNT = "ninadnar98-z0upz59iv"
dataset_id, path = sys.argv[1], sys.argv[2]
key = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))["FIREWORKS_API_KEY"]
c = Fireworks(api_key=key, account_id=ACCOUNT)
rows = sum(1 for _ in open(path))
name = os.path.basename(path)
size = os.path.getsize(path)

try:
    d = c.datasets.get(dataset_id)
    print("exists:", d.state)
except Exception:
    d = c.datasets.create(dataset_id=dataset_id, dataset={"userUploaded": {}, "exampleCount": str(rows)})
    print("created:", d.name, d.state)

ep = c.datasets.get_upload_endpoint(dataset_id, filename_to_size={name: str(size)})
urls = getattr(ep, "filename_to_signed_urls", None) or ep.model_dump().get("filenameToSignedUrls") or ep.model_dump().get("filename_to_signed_urls")
url = urls[name]
print(f"uploading {name} ({size/1e6:.1f} MB, {rows} rows)...")
with open(path, "rb") as fh:
    r = httpx.put(url, content=fh, headers={"Content-Type": "application/octet-stream",
                                             "x-goog-content-length-range": f"{size},{size}"}, timeout=1800)
print("upload HTTP", r.status_code, r.text[:200])
r.raise_for_status()
c.datasets.validate_upload(dataset_id, body={})
for _ in range(120):
    d = c.datasets.get(dataset_id)
    if str(d.state).endswith("READY") or "FAIL" in str(d.state):
        break
    time.sleep(10)
print("final:", d.state, "rows:", getattr(d, "example_count", None), getattr(d, "status", None))

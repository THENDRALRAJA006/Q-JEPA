import urllib.request
import json

url = "https://zenodo.org/api/records/5724362"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
        print("Title:", data.get("metadata", {}).get("title"))
        print("License:", data.get("metadata", {}).get("license"))
        print("Description:", data.get("metadata", {}).get("description")[:200])
        print("Files:")
        for f in data.get("files", []):
            size_mb = f.get("size", 0) / (1024 * 1024)
            key = f.get("key")
            download_url = f.get("links", {}).get("self")
            print(f"  {key} ({size_mb:.2f} MB): {download_url}")
except Exception as e:
    print("Zenodo API error:", e)

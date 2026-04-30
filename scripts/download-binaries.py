# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.32",
# ]
# ///

import argparse
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Dict

PACKAGE_MAPPING: Dict[str, str] = {
    "Faiss.NET.Native.Linux":                "src/FaissNET.Linux",
    "Faiss.NET.Native.Windows":              "src/FaissNET.Windows",
    "Faiss.NET.Native.MacOS":                "src/FaissNET.MacOS",
    "Faiss.NET.Gpu.Native.Cuda.Linux.x64":   "src/FaissNET.Linux.Gpu.Cuda.x64",
    "Faiss.NET.Gpu.Native.Cuda.Linux.arm64": "src/FaissNET.Linux.Gpu.Cuda.arm64",
    "Faiss.NET.Gpu.Native.Rocm.Linux":       "src/FaissNET.Linux.Gpu.Rocm",
}

BASE_URL = "https://api.nuget.org/v3-flatcontainer"

def download_and_extract_natives(version: str, root_dir: Path):
    for package_name, rel_target_dir in PACKAGE_MAPPING.items():
        target_dir = root_dir / rel_target_dir / "runtimes"
        if not target_dir.exists():
            print(f"⚠️  Target directory not found: {target_dir}")
            continue

        pkg_lower = package_name.lower()
        url = f"{BASE_URL}/{pkg_lower}/{version}/{pkg_lower}.{version}.nupkg"

        print(f"Downloading: {package_name}.{version}.nupkg")

        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            print(f"   ❌ Package not found on nuget.org (404) → {package_name}")
            continue
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".nupkg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

        with zipfile.ZipFile(tmp_path) as z:
            extracted_count = 0
            for member in z.namelist():
                if member.startswith("runtimes/") and "/native/" in member:
                    z.extract(member, target_dir.parent)
                    print(f"   ✓ Extracted {member}")
                    extracted_count += 1

            if extracted_count == 0:
                print(f"   ⚠️  No native files found in package")

        tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Download Faiss.NET runtime NuGet packages from nuget.org and update local native libraries."
    )
    parser.add_argument("-v", "--version", default="1.0.0-preview.4", help="NuGet package version (e.g. 1.0.0-preview.2)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent

    print(f"Updating native libraries for version {args.version} from nuget.org\n")
    download_and_extract_natives(args.version, root_dir)
    print("\n✅ Done! Native libraries updated.")


if __name__ == "__main__":
    main()
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.32",
# ]
# ///

import argparse
import platform
import sys
from pathlib import Path
import zipfile
import io
import requests
from typing import Dict

# ====================== CONFIGURATION ======================
# Update this dict with your exact runtime package base names
# (look at https://github.com/Binozo?tab=packages&repo_name=Faiss.NET)
RUNTIME_PACKAGES: Dict[str, str] = {
    "win-x64":    "Faiss.NET.runtime.win-x64",
    "linux-x64":  "Faiss.NET.runtime.linux-x64",
    "linux-arm64": "Faiss.NET.runtime.linux-arm64",
    "osx-x64":    "Faiss.NET.runtime.osx-x64",
    "osx-arm64":  "Faiss.NET.runtime.osx-arm64",
    # Add any other RIDs you publish (e.g. linux-musl-x64, etc.)
}

# Set to "" if your release tags are just the version number (e.g. "1.8.0")
RELEASE_TAG_PREFIX = "v"
# ===========================================================

def get_current_rid() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "win-x64" if machine in ("amd64", "x86_64") else "win-x86"
    elif system == "darwin":
        return "osx-arm64" if machine in ("arm64", "aarch64") else "osx-x64"
    elif system == "linux":
        if machine in ("aarch64", "arm64"):
            return "linux-arm64"
        return "linux-x64"
    else:
        raise RuntimeError(f"Unsupported platform: {system}-{machine}")


def download_and_extract(version: str, rid: str, output_base: Path) -> bool:
    pkg_base = RUNTIME_PACKAGES.get(rid)
    if not pkg_base:
        print(f"⚠️  No package defined for RID '{rid}'")
        return False

    nupkg_name = f"{pkg_base}.{version}.nupkg"
    tag = f"{RELEASE_TAG_PREFIX}{version}" if RELEASE_TAG_PREFIX else version
    url = f"https://github.com/Binozo/Faiss.NET/releases/download/{tag}/{nupkg_name}"

    print(f"→ Downloading {nupkg_name} ({rid})...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to download {rid}: {e}")
        return False

    target_dir = output_base / rid / "native"
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        extracted = 0
        for member in z.namelist():
            if f"runtimes/{rid}/native/" in member and member.endswith((".dll", ".so", ".dylib")):
                z.extract(member, output_base)
                print(f"   ✓ Extracted {Path(member).name}")
                extracted += 1

        if extracted == 0:
            print(f"   ⚠️  No native library found for {rid}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Faiss.NET prebuilt native runtimes from GitHub Releases"
    )
    parser.add_argument("--version", required=True, help="Version without 'v' (e.g. 1.8.0)")
    parser.add_argument("--all", action="store_true", help="Download for ALL platforms instead of current one")
    parser.add_argument("-o", "--output", default="runtimes", help="Base output directory (default: runtimes)")
    args = parser.parse_args()

    output_base = Path(args.output)

    if args.all:
        rids = list(RUNTIME_PACKAGES.keys())
        print(f"Downloading Faiss runtimes for all {len(rids)} platforms...\n")
    else:
        try:
            rid = get_current_rid()
            rids = [rid]
            print(f"Downloading Faiss runtime for current platform ({rid})...\n")
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    success = 0
    for rid in rids:
        if download_and_extract(args.version, rid, output_base):
            success += 1

    print(f"\n✅ Finished! Successfully processed {success}/{len(rids)} platforms.")


if __name__ == "__main__":
    main()

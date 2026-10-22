"""Hub dataset files cached on local disk under a size budget, shared by all DataLoader workers."""

import os
import shutil
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download


class HubCache:
    """Least-recently-used file cache for Hub dataset repos.

    Downloads go to a private staging dir and are moved in with an atomic rename, so other workers never see a
    partial file. A cache hit refreshes the file's mtime, which is the LRU clock; eviction tolerates files that
    another worker deletes at the same time. A reader can still lose a file evicted between lookup and open; the
    caller treats that like any other failed window and samples a new one.
    """

    def __init__(self, root: str | Path, max_bytes: float, prune_every: int = 500):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.prune_every = prune_every
        self._downloads = 0

    def get(self, repo: str, filename: str) -> Path:
        path = self.root / repo / filename
        if path.exists():
            os.utime(path)
            return path
        staging = self.root / ".staging" / uuid.uuid4().hex
        try:
            fetched = hf_hub_download(repo, filename, repo_type="dataset", local_dir=staging)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(fetched, path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self._downloads += 1
        if self._downloads % self.prune_every == 0:
            self.prune()
        return path

    def prune(self) -> None:
        """Delete least recently used files until the cache fits in max_bytes."""
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d != ".staging"]
            for name in filenames:
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except FileNotFoundError:
                    continue
                files.append((st.st_mtime, st.st_size, os.path.join(dirpath, name)))
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.max_bytes:
                break
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            total -= size

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


class LocalStorage:
    def __init__(self) -> None:
        self.root = settings.storage_dir

    def scan_dir(self, scan_id: str) -> Path:
        safe_scan_id = scan_id.strip() or uuid.uuid4().hex
        directory = self.root / "scans" / safe_scan_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    async def save_upload(self, scan_id: str, upload: UploadFile, max_bytes: int) -> Path:
        target_dir = self.scan_dir(scan_id)
        suffix = Path(upload.filename or "certificate").suffix.lower() or ".bin"
        target_path = target_dir / f"input{suffix}"
        written = 0
        upload.file.seek(0)
        with target_path.open("wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise ValueError("Upload exceeds the configured size limit.")
                buffer.write(chunk)
        await upload.close()
        return target_path

    def save_artifact(self, source: Path, scan_id: str, name: str) -> Path:
        target_dir = self.scan_dir(scan_id)
        target_path = target_dir / name
        if source != target_path:
            shutil.copy2(source, target_path)
        return target_path

    def artifact_url(self, path: Path) -> str:
        relative_path = path.relative_to(self.root).as_posix()
        return f"{settings.static_url_prefix}/{relative_path}"

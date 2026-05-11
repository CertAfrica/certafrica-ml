from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import Config

from app.config import settings


class SupabaseS3Storage:
    def __init__(self) -> None:
        if not settings.s3_bucket_name:
            raise RuntimeError("S3 bucket name is not configured.")
        if not settings.s3_endpoint_url:
            raise RuntimeError("S3 endpoint URL is not configured.")

        self.bucket_name = settings.s3_bucket_name
        self.endpoint_url = settings.s3_endpoint_url.rstrip("/")
        self.public_base_url = settings.s3_public_base_url or self._derive_public_base_url(self.endpoint_url)
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region_name,
            config=Config(signature_version="s3v4"),
        )

    def object_key(self, scan_id: str, name: str) -> str:
        safe_scan_id = scan_id.strip() or uuid.uuid4().hex
        safe_name = Path(name).name
        return f"scans/{safe_scan_id}/{safe_name}"

    def upload_file(self, source: Path, key: str, content_type: str | None = None) -> str:
        inferred_content_type = content_type or mimetypes.guess_type(str(source))[0] or "application/octet-stream"
        extra_args = {"ContentType": inferred_content_type}
        with source.open("rb") as buffer:
            self.client.upload_fileobj(buffer, self.bucket_name, key, ExtraArgs=extra_args)
        return self.public_url(key)

    def public_url(self, key: str) -> str:
        normalized_key = key.lstrip("/")
        return f"{self.public_base_url.rstrip('/')}/{self.bucket_name}/{normalized_key}"

    def _derive_public_base_url(self, endpoint_url: str) -> str:
        parsed = urlparse(endpoint_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return f"{base}/storage/v1/object/public"

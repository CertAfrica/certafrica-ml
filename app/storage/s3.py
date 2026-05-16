from __future__ import annotations

import mimetypes
from dataclasses import dataclass
import uuid
from pathlib import Path
from urllib.parse import urlparse, quote
import os
import logging
from typing import List, Optional

import boto3
from botocore.client import Config

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedObject:
    key: str
    path: Path


class SupabaseS3Storage:
    def __init__(self) -> None:
        if not settings.s3_bucket_name:
            raise RuntimeError("S3 bucket name is not configured.")
        if not settings.s3_endpoint_url:
            raise RuntimeError("S3 endpoint URL is not configured.")

        self.bucket_name = settings.s3_bucket_name
        self.endpoint_url = settings.s3_endpoint_url.rstrip("/")
        self.public_base_url = settings.s3_public_base_url or self._derive_public_base_url(self.endpoint_url)
        self.signed_urls_enabled = settings.s3_signed_urls_enabled
        self.signed_url_expires_seconds = settings.s3_signed_url_expires_seconds
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

    def scan_prefix(self, scan_id: str) -> str:
        safe_scan_id = scan_id.strip() or uuid.uuid4().hex
        return f"scans/{safe_scan_id}/"

    def list_keys(self, prefix: str) -> List[str]:
        logger.info('Listing keys under scan prefix', extra={"prefix": prefix})
        paginator = self.client.get_paginator('list_objects_v2')
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        logger.info('Listed keys under scan prefix', extra={"prefix": prefix, "found": len(keys)})
        return keys

    def find_scan_certificate_key(self, scan_id: str) -> Optional[str]:
        prefix = self.scan_prefix(scan_id)
        logger.info('Searching for certificate key for scan', extra={"scan_id": scan_id, "prefix": prefix})
        keys = self.list_keys(prefix)
        for k in keys:
            base = os.path.basename(k).lower()
            if base.startswith('certificate'):
                logger.info('Found certificate key for scan', extra={"scan_id": scan_id, "key": k})
                return k
        logger.warning('No certificate key found for scan', extra={"scan_id": scan_id, "prefix": prefix})
        return None

    def download_to_path(self, key: str, target: Path) -> DownloadedObject:
        logger.info('Downloading object from S3 to local path', extra={"key": key, "target": str(target)})
        try:
            self.client.download_file(self.bucket_name, key.lstrip("/"), str(target))
        except Exception as e:
            logger.error('Failed to download object from S3', extra={"key": key, "target": str(target), "err": str(e)})
            raise
        logger.info('Downloaded object from S3', extra={"key": key, "target": str(target)})
        return DownloadedObject(key=key.lstrip("/"), path=target)

    def upload_file(self, source: Path, key: str, content_type: str | None = None) -> str:
        logger.info('Uploading artifact to S3', extra={"local_path": str(source), "remote_key": key})
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        try:
            with source.open("rb") as buffer:
                self.client.upload_fileobj(buffer, self.bucket_name, key, ExtraArgs=extra_args)
        except Exception as e:
            logger.error('Failed to upload artifact to S3', extra={"local_path": str(source), "remote_key": key, "err": str(e)})
            raise
        url = self.object_url(key)
        logger.info('Uploaded artifact to S3', extra={"remote_key": key, "url": url})
        return url

    def object_url(self, key: str) -> str:
        if self.signed_urls_enabled:
            return self.signed_url(key)
        return self.public_url(key)

    def signed_url(self, key: str, expires_in: int | None = None) -> str:
        normalized_key = key.lstrip("/")
        ttl_seconds = expires_in if expires_in is not None else self.signed_url_expires_seconds
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": normalized_key},
            ExpiresIn=ttl_seconds,
        )

    def public_url(self, key: str) -> str:
        normalized_key = key.lstrip("/")
        return f"{self.public_base_url.rstrip('/')}/{self.bucket_name}/{normalized_key}"

    def _derive_public_base_url(self, endpoint_url: str) -> str:
        parsed = urlparse(endpoint_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return f"{base}/storage/v1/object/public"

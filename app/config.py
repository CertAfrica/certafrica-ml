from pathlib import Path
from typing import Any, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CertAfrica ML Service"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    storage_dir: Path = Path("storage")
    uploads_dir: Path = Path("storage/uploads")
    heatmaps_dir: Path = Path("storage/heatmaps")
    reports_dir: Path = Path("storage/reports")
    max_upload_mb: int = 15
    ocr_lang: str = "eng"
    static_url_prefix: str = "/artifacts"
    s3_endpoint_url: str = "https://xshrsqhjbwlldvshtjcr.storage.supabase.co/storage/v1/s3"
    s3_bucket_name: str = "certafrica"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region_name: str = "eu-central-1"
    s3_public_base_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def model_post_init(self, __context: Any) -> None:
        legacy_values = self._load_legacy_supabase_values(Path(".env"))
        if not self.s3_endpoint_url:
            self.s3_endpoint_url = legacy_values.get("endpoint_url", self.s3_endpoint_url)
        if not self.s3_access_key_id:
            self.s3_access_key_id = legacy_values.get("access_key_id", self.s3_access_key_id)
        if not self.s3_secret_access_key:
            self.s3_secret_access_key = legacy_values.get("secret_access_key", self.s3_secret_access_key)
        if not self.s3_public_base_url and self.s3_endpoint_url:
            self.s3_public_base_url = self._derive_public_base_url(self.s3_endpoint_url)

    def _load_legacy_supabase_values(self, env_path: Path) -> Dict[str, str]:
        if not env_path.exists():
            return {}

        bare_lines = []
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" in stripped:
                continue
            bare_lines.append(stripped)

        values: Dict[str, str] = {}
        if bare_lines:
            values["access_key_id"] = bare_lines[0]
        if len(bare_lines) > 1:
            values["secret_access_key"] = bare_lines[1]
        for item in reversed(bare_lines):
            if "storage/v1/s3" in item:
                values["endpoint_url"] = item
                break
        return values

    def _derive_public_base_url(self, endpoint_url: str) -> str:
        return endpoint_url.replace("/storage/v1/s3", "/storage/v1/object/public").rstrip("/")

    def ensure_directories(self) -> None:
        for directory in (self.storage_dir / "ai",):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

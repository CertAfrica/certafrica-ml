# CertAfrica ML Service

A lightweight FastAPI service for Nigerian certificate verification. It analyzes uploaded certificates with practical OCR and image-forensics heuristics, then returns a trust score, verification status, heatmap, and downloadable PDF report.

## What it does
- Validates PNG, JPG, JPEG, and PDF uploads
- Preprocesses images with grayscale conversion, resizing, denoising, and normalization
- Extracts OCR text and structured fields
- Runs explainable forgery checks with ELA, contour analysis, OCR confidence checks, and duplicate-region heuristics
- Generates a visual heatmap and a PDF verification report
- Uploads uploads, previews, heatmaps, and reports to Supabase S3-compatible storage

## API
### `POST /analyze`
Form fields:
- `scan_id`: string

Response includes:
- `scanId`
- `analysisTimestamp`
- `processingTime`
- `verification`
  - `trustScore`
  - `refinedTrustScore`
  - `riskLevel`
  - `confidenceLevel`
  - `status`
  - `recommendedAction`
- `documentAnalysis` (from AI layer)
  - `documentType`
  - `certificateType`
  - `institution`
- `ocrAnalysis`
  - `confidence`
  - `language`
  - `rawText`
  - `extractedFields`
- `anomalyDetection`
  - `overallScore`
  - `findingSummary`
  - `suspiciousRegions` (high-severity only)
- `layoutAnalysis`
  - `hasSignatureArea`
  - `hasTableStructure`
  - `hasHeaderArea`
  - `hasFooterArea`
  - `blockCount`
- `reasoning`
- `artifacts`
  - `uploadUrl`
  - `heatmapUrl`
  - `reportUrl`
  - `previewUrl`

### `GET /health`
Simple readiness endpoint.

## Folder structure
```text
app/
	api/
	detection/
	heatmaps/
	models/
	preprocessing/
	reports/
	services/
	storage/
	utils/
```

## Dependencies
- `FastAPI`
- `OpenCV`
- `NumPy`
- `Pillow`
- `scikit-learn`
- `PyMuPDF`
- `ReportLab`
- `boto3`

## Local setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

> Note: OCR runs through EasyOCR, so you do not need the `tesseract` binary.

## Docker
```bash
docker build -t certafrica-ml .
docker run -p 8000:8000 certafrica-ml
```

Or use Compose:
```bash
docker compose up --build
```

## Environment variables
- `APP_NAME`
- `ENVIRONMENT`
- `API_HOST`
- `API_PORT`
- `STORAGE_DIR`
- `MAX_UPLOAD_MB`
- `OCR_LANG`
- `STATIC_URL_PREFIX`
- `S3_ENDPOINT_URL`
- `S3_BUCKET_NAME` (should match the backend bucket)
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_REGION_NAME`
- `S3_PUBLIC_BASE_URL`
- `S3_SIGNED_URLS_ENABLED` (default: `true`)
- `S3_SIGNED_URL_EXPIRES_SECONDS` (default: `3600`)

## Development roadmap
1. Add a small certificate dataset for tuning thresholds.
2. Replace heuristic OCR parsing with better field extraction if needed.
3. Tune Supabase signed URL TTL and bucket policies for production.
4. Add request auth and rate limiting when exposed publicly.
5. Add regression tests for upload validation and scoring.

## Notes
This service is intentionally simple and hackathon-friendly. It favors explainable, fast heuristics over heavyweight deep learning.

# CertAfrica AI Layer

The AI layer complements the ML service by interpreting OCR and anomaly outputs rather than replacing the core vision pipeline.

## AI layer API
- `POST /ai/interpret` — refine ML output into certificate type, institution, scoring, and explanation
- `GET /ai/health` — readiness check for the interpreter
- `POST /ai/train/sample` — tiny demo training hook for the certificate type classifier

## AI layer responsibilities
- Recognize certificate types with rules plus a lightweight trainable text model
- Infer institutions with fuzzy matching, OCR cleanup, and a small knowledge base
- Refine trust score with explainable scoring rules
- Convert suspicious regions into human-readable anomaly notes
- Produce concise, human-readable verification summaries

## AI folder structure
```text
app/ai/
	api/
	classifiers/
	explainers/
	knowledge_base/
	scoring/
	services/
	training/
	utils/
```

## AI knowledge base
The bundled knowledge base lives in `app/ai/knowledge_base/knowledge_base.json`. On first run it is copied into `storage/ai/knowledge_base.json` so you can update runtime patterns without touching the source seed.

## AI development roadmap
1. Add more labeled certificate examples for the lightweight text classifier.
2. Expand institution aliases for Nigerian schools and exam bodies.
3. Tune trust refinement thresholds from real review feedback.
4. Add a small curated set of certificate layout templates.
5. Keep the knowledge base in sync with verified institutions.
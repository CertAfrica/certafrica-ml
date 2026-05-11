from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Flowable, Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models.schemas import AnomalyResult, OCRResult


class ReportGenerator:
    def generate(
        self,
        output_path: Path,
        scan_id: str,
        trust_score: int,
        verification_status: str,
        ocr_result: OCRResult,
        anomaly_result: AnomalyResult,
        certificate_image_path: Path,
        heatmap_image_path: Path,
        created_at: datetime,
        ai_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ai_context = ai_context or {}
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=1.2 * cm, leftMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
        story: List[Flowable] = []

        document_type = self._text(ai_context, "documentType.display_name", "Unknown Document")
        certificate_type = self._text(ai_context, "certificateType.display_name", "Unknown Document")
        institution = self._text(ai_context, "institution.canonical_name", "Unknown Institution")
        refined_score = self._text(ai_context, "refinedTrustScore", str(trust_score))
        risk_level = self._text(ai_context, "riskLevel", verification_status)
        confidence_level = self._text(ai_context, "confidenceLevel", "unknown")
        summary_text = self._text(ai_context, "verificationSummary", verification_status.replace("_", " ").title())
        detailed_explanation = self._text(ai_context, "detailedExplanation", "No AI explanation available.")
        recommended_action = self._text(ai_context, "recommendedAction", "Review manually.")
        ai_reasoning = ai_context.get("reasoning", []) or []
        ai_notes = ai_context.get("anomalyNotes", []) or []
        ai_metadata = ai_context.get("metadata", {}) or {}

        title_table = Table(
            [[Paragraph("CertAfrica Verification Report", styles["Title"]), Paragraph(f"Scan ID: {scan_id}", styles["Heading3"]) ]],
            colWidths=[11.5 * cm, 4.5 * cm],
        )
        title_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f4c81")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#0f4c81")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(title_table)
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph(f"Generated: {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        summary_data = [
            ["Trust Score", f"{trust_score}/100"],
            ["Status", verification_status.replace("_", " ").title()],
            ["Document Type", document_type],
            ["Certificate Type", certificate_type],
            ["Institution", institution],
            ["AI Refined Score", f"{refined_score}/100"],
            ["AI Risk Level", str(risk_level).replace("_", " ").title()],
            ["AI Confidence", str(confidence_level).title()],
            ["OCR Confidence", f"{ocr_result.average_confidence:.1f}%"],
            ["Anomaly Score", f"{anomaly_result.overall_score:.1f}/100"],
            ["Recommended Action", recommended_action],
        ]
        summary_table = Table(summary_data, colWidths=[5.0 * cm, 9.0 * cm])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceeff")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(summary_table)
        story.append(Spacer(1, 0.35 * cm))

        story.append(Paragraph("AI Verdict", styles["Heading2"]))
        story.append(Paragraph(self._escape_multiline(summary_text), styles["BodyText"]))
        story.append(Paragraph(self._escape_multiline(detailed_explanation), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))

        if ai_reasoning:
            story.append(Paragraph("AI Reasoning", styles["Heading2"]))
            for reason in ai_reasoning[:6]:
                story.append(Paragraph(f"• {self._escape_multiline(str(reason))}", styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))

        if ai_notes:
            story.append(Paragraph("AI Anomaly Notes", styles["Heading2"]))
            for note in ai_notes[:8]:
                story.append(Paragraph(f"• {self._escape_multiline(str(note))}", styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))

        if ai_metadata:
            story.append(Paragraph("AI Metadata", styles["Heading2"]))
            metadata_rows = [[key.replace("_", " ").title(), self._format_value(value)] for key, value in ai_metadata.items()]
            metadata_table = Table(metadata_rows, colWidths=[5.0 * cm, 9.0 * cm])
            metadata_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.grey), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
            story.append(metadata_table)
            story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("OCR Extracted Text", styles["Heading2"]))
        story.append(Paragraph(self._escape_multiline(ocr_result.raw_text or "No readable text detected."), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph("Structured OCR Fields", styles["Heading2"]))
        fields = ocr_result.extracted_fields or {}
        field_rows = [[key.replace("_", " ").title(), value] for key, value in fields.items()] or [["Result", "No structured fields extracted."]]
        field_table = Table(field_rows, colWidths=[5.0 * cm, 9.0 * cm])
        field_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.grey), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
        story.append(field_table)
        story.append(Spacer(1, 0.35 * cm))

        story.append(Paragraph("Anomaly Findings", styles["Heading2"]))
        findings = anomaly_result.findings or ["No major anomalies detected."]
        for finding in findings:
            story.append(Paragraph(f"• {self._escape_multiline(finding)}", styles["BodyText"]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Suspicious Regions", styles["Heading2"]))
        region_rows = [["Region", "Severity", "AI Note"]]
        for index, region in enumerate(anomaly_result.suspicious_regions or []):
            region_rows.append(
                [
                    f"{index + 1}: {self._region_summary(region)}",
                    f"{float(region.get('severity', 0.0) or 0.0):.1f}",
                    self._format_value(region.get("ai_notes") or "No AI note available."),
                ]
            )
        region_table = Table(region_rows, colWidths=[7.0 * cm, 2.5 * cm, 4.5 * cm])
        region_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1ff")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(region_table)
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Visual Evidence", styles["Heading2"]))
        visuals = Table(
            [[
                self._image_or_placeholder(certificate_image_path, max_width=7.6 * cm, styles=styles),
                self._image_or_placeholder(heatmap_image_path, max_width=7.6 * cm, styles=styles),
            ]],
            colWidths=[8.0 * cm, 8.0 * cm],
        )
        visuals.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(visuals)

        doc.build(story)
        return output_path

    def _image_or_placeholder(self, image_path: Path, max_width: float, styles) -> Flowable:
        if not image_path.exists():
            return Paragraph(f"<i>Image unavailable: {self._escape(str(image_path))}</i>", styles["BodyText"])
        return self._scaled_image(image_path, max_width=max_width)

    def _scaled_image(self, image_path: Path, max_width: float) -> RLImage:
        image = RLImage(str(image_path))
        ratio = max_width / image.drawWidth if image.drawWidth else 1.0
        image.drawWidth = max_width
        image.drawHeight = image.drawHeight * ratio
        return image

    def _escape(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _escape_multiline(self, text: str) -> str:
        return self._escape(text).replace("\n", "<br/>")

    def _text(self, data: Dict[str, Any], path: str, default: str) -> str:
        current: Any = data
        for part in path.split("."):
            if not isinstance(current, dict):
                return default
            current = current.get(part)
            if current is None:
                return default
        return str(current)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return self._escape_multiline(str(value))
        return self._escape_multiline(str(value))

    def _region_summary(self, region: Dict[str, Any]) -> str:
        left = int(region.get("left", 0) or 0)
        top = int(region.get("top", 0) or 0)
        width = int(region.get("width", 0) or 0)
        height = int(region.get("height", 0) or 0)
        return f"x={left}, y={top}, w={width}, h={height}"

#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Template:
    category: str
    subcategory: str
    urgency: str
    escalation_probability: float
    edge_case: bool
    inquiry_template: str


PROGRAMS = [
    "Data Science & AI",
    "Industry 4.0 Management",
    "Digital Business & Sustainability",
    "Digital Marketing & Social Media Bachelor Program",
    "Coding & Software Engineering Bachelors Program",
    "Data Science MSc.",
    "Industry 4.0 MSc.",
    "International MBA",
    "Digital Transformation & Sustainability M.A."
]

INTAKES = ["April 2026", "October 2026", "April 2027", "October 2027"]
PAYMENT_METHODS = ["bank transfer", "credit card", "SEPA direct debit", "installment plan"]
CHANNELS = ["email", "chat", "phone", "portal"]
LANGUAGES = ["en", "zh",  "de", "es", "fr", "ar", "ru", "hi", "pt", "ja"]

INQUIRY_TEXT_VARIANTS = [
    " Please share the exact next steps.",
    " Kindly confirm the required documents and timeline.",
    " A brief checklist would help.",
    " Please advise the fastest available process.",
    " Could you confirm this in writing?",
    " I would appreciate a clear deadline.",
    " Please include who to contact next.",
    " Let me know if any additional proof is needed.",
    " Please also mention expected processing time.",
    " I need this clarified before {intake} starts.",
    " Please respond through the {channel} channel if possible.",
    " A confirmation message today would be very helpful.",
    " Please include any fee or policy details that apply.",
    " I want to avoid delays, so a precise answer would help.",
]


HIGH_FREQUENCY_TEMPLATES: list[Template] = [
    Template("enrollment", "application_status", "medium", 0.20, False, "I applied for {program} for {intake}. Could you confirm my application status and next steps?"),
    Template("enrollment", "document_submission", "medium", 0.25, False, "I submitted my passport and transcripts for {program}, but the portal still shows missing documents. Can you check?"),
    Template("enrollment", "admission_requirements", "low", 0.15, False, "What are the admission requirements for {program} if I am applying from outside Germany?"),
    Template("enrollment", "offer_letter", "high", 0.35, False, "I need my offer letter for visa processing. When will it be issued for {intake}?"),
    Template("payments", "invoice_request", "medium", 0.25, False, "Could you send me the tuition invoice for {program} starting {intake}?"),
    Template("payments", "payment_confirmation", "medium", 0.30, False, "I paid by {payment_method} three days ago but have not received confirmation. Can you verify receipt?"),
    Template("payments", "installment_plan", "medium", 0.35, False, "Can I switch to an installment plan for my tuition payment this semester?"),
    Template("payments", "refund_request", "high", 0.55, False, "My enrollment changed and I need to request a refund. What is the process and timeline?"),
    Template("deadlines", "application_deadline", "medium", 0.20, False, "What is the final application deadline for {program} for {intake}?"),
    Template("deadlines", "document_deadline", "high", 0.40, False, "I may miss the document deadline by 2 days due to courier delays. Is an extension possible?"),
    Template("deadlines", "payment_deadline", "high", 0.45, False, "I received a payment reminder. What happens if tuition is paid after the deadline?"),
    Template("certificates", "enrollment_certificate", "medium", 0.20, False, "I need an enrollment certificate for my residence permit appointment. How can I get it quickly?"),
    Template("certificates", "transcript_request", "medium", 0.25, False, "Please issue my transcript in English with ECTS credits for transfer applications."),
    Template("certificates", "graduation_documents", "low", 0.15, False, "When will graduation certificates be available after final grades are published?"),
]


EDGE_CASE_TEMPLATES: list[Template] = [
    Template("enrollment", "identity_mismatch", "high", 0.80, True, "My legal name changed after application, and now my passport and portal account names do not match. Please advise urgently."),
    Template("enrollment", "duplicate_application", "medium", 0.65, True, "I accidentally submitted two applications under different emails for {program}. Can you merge them?"),
    Template("payments", "double_charge", "high", 0.85, True, "I was charged tuition twice this week. I need immediate correction and refund confirmation."),
    Template("payments", "fraud_alert", "critical", 0.95, True, "I received a suspicious payment link claiming to be from the university. Please confirm if it is legitimate."),
    Template("deadlines", "medical_exception", "high", 0.75, True, "I was hospitalized and missed the enrollment deadline. Can my case be reviewed as an exception?"),
    Template("deadlines", "visa_delay", "high", 0.70, True, "My visa interview was postponed by the embassy and I cannot arrive before semester start. What are my options?"),
    Template("certificates", "urgent_embassy_letter", "critical", 0.90, True, "My embassy appointment is tomorrow and I urgently need a stamped enrollment confirmation today."),
    Template("certificates", "grade_dispute", "high", 0.80, True, "My transcript shows an incorrect grade for one module. I want a formal review process now."),
    Template("compliance", "data_privacy_request", "medium", 0.75, True, "I want to request all personal data held by the university and ask for deletion of non-essential records."),
    Template("legal", "complaint_escalation", "critical", 0.98, True, "I filed two unresolved complaints regarding tuition handling. Please escalate to management immediately."),
    Template("accessibility", "disability_accommodation", "high", 0.70, True, "I need exam accommodations due to a documented disability. Who handles this and what documents are required?"),
]


def escalation_band(prob: float) -> str:
    if prob >= 0.80:
        return "very_high"
    if prob >= 0.60:
        return "high"
    if prob >= 0.35:
        return "medium"
    return "low"


def synthesize_record(index: int, template: Template, rng: random.Random) -> dict[str, Any]:
    program = rng.choice(PROGRAMS)
    intake = rng.choice(INTAKES)
    payment_method = rng.choice(PAYMENT_METHODS)
    channel = rng.choice(CHANNELS)
    language = rng.choice(LANGUAGES)

    variant_tail = rng.choice(INQUIRY_TEXT_VARIANTS)
    inquiry_text = (
        template.inquiry_template.format(
            program=program,
            intake=intake,
            payment_method=payment_method,
        )
        + variant_tail.format(
            program=program,
            intake=intake,
            payment_method=payment_method,
            channel=channel,
            language=language,
        )
    )

    # Light random jitter keeps rows diverse while preserving template intent.
    jitter = rng.uniform(-0.06, 0.08)
    escalation_probability = min(0.99, max(0.01, template.escalation_probability + jitter))

    record = {
        "inquiry_id": f"SYN-{index:04d}",
        "category": template.category,
        "subcategory": template.subcategory,
        "urgency": template.urgency,
        "escalation_probability": round(escalation_probability, 2),
        "escalation_band": escalation_band(escalation_probability),
        "edge_case": template.edge_case,
        "channel": channel,
        "language": language,
        "intake": intake,
        "program": program,
        "inquiry_text": inquiry_text,
    }
    return record


def synthesize_unique_record(
    index: int,
    template: Template,
    rng: random.Random,
    used_texts: set[str],
    max_attempts: int = 200,
) -> dict[str, Any]:
    for _ in range(max_attempts):
        record = synthesize_record(index=index, template=template, rng=rng)
        text = str(record["inquiry_text"]).strip()
        if text in used_texts:
            continue
        used_texts.add(text)
        return record

    raise RuntimeError(
        f"Unable to generate unique inquiry_text after {max_attempts} attempts "
        f"for template {template.category}/{template.subcategory}."
    )


def generate_dataset(total: int = 200, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    used_texts: set[str] = set()

    # Keep high-frequency cases dominant, with a meaningful edge-case ratio.
    edge_target = max(30, int(total * 0.22))
    regular_target = total - edge_target

    rows: list[dict[str, Any]] = []

    for i in range(regular_target):
        tmpl = HIGH_FREQUENCY_TEMPLATES[i % len(HIGH_FREQUENCY_TEMPLATES)]
        rows.append(synthesize_unique_record(i + 1, tmpl, rng, used_texts))

    for i in range(edge_target):
        tmpl = EDGE_CASE_TEMPLATES[i % len(EDGE_CASE_TEMPLATES)]
        rows.append(synthesize_unique_record(regular_target + i + 1, tmpl, rng, used_texts))

    rng.shuffle(rows)

    # Re-number after shuffle for clean ordering.
    for idx, row in enumerate(rows, start=1):
        row["inquiry_id"] = f"SYN-{idx:04d}"

    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_category: dict[str, int] = {}
    by_urgency: dict[str, int] = {}
    edge_count = 0

    for row in rows:
        cat = str(row["category"])
        urg = str(row["urgency"])
        by_category[cat] = by_category.get(cat, 0) + 1
        by_urgency[urg] = by_urgency.get(urg, 0) + 1
        if bool(row["edge_case"]):
            edge_count += 1

    summary = {
        "total_records": len(rows),
        "edge_case_records": edge_count,
        "high_frequency_records": len(rows) - edge_count,
        "category_distribution": by_category,
        "urgency_distribution": by_urgency,
        "schema": [
            "inquiry_id",
            "category",
            "subcategory",
            "urgency",
            "escalation_probability",
            "escalation_band",
            "edge_case",
            "channel",
            "language",
            "intake",
            "program",
            "inquiry_text",
        ],
    }

    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "test" / "fixtures"

    rows = generate_dataset(total=300, seed=42)

    save_jsonl(out_dir / "synthetic_inquiries.jsonl", rows)
    save_csv(out_dir / "synthetic_inquiries.csv", rows)
    save_summary(out_dir / "synthetic_inquiries_summary.json", rows)

    print(f"Generated {len(rows)} synthetic inquiries")
    print(f"Saved: {out_dir / 'synthetic_inquiries.jsonl'}")
    print(f"Saved: {out_dir / 'synthetic_inquiries.csv'}")
    print(f"Saved: {out_dir / 'synthetic_inquiries_summary.json'}")


if __name__ == "__main__":
    main()

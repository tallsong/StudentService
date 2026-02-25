#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Playbook:
    title: str
    short_answer: str
    required_documents: list[str]
    process_steps: list[str]
    expected_timeline: str
    escalation_triggers: list[str]
    default_owner: str


PLAYBOOKS: dict[str, Playbook] = {
    "enrollment.application_status": Playbook(
        title="Application status check",
        short_answer="We verify your application stage, confirm missing items, and share the next milestone.",
        required_documents=["application ID", "registered email"],
        process_steps=[
            "Confirm identity using application ID and registered email.",
            "Check the latest workflow state in the admissions portal.",
            "List pending items or validation blockers.",
            "Send the next action and expected review date.",
        ],
        expected_timeline="1-3 business days",
        escalation_triggers=["No status update after 5 business days", "Conflicting portal states"],
        default_owner="Admissions",
    ),
    "enrollment.document_submission": Playbook(
        title="Document receipt mismatch",
        short_answer="We reconcile uploaded files with the checklist and manually relink missing documents.",
        required_documents=["application ID", "upload timestamp", "document names"],
        process_steps=[
            "Confirm the document upload timestamp and file names.",
            "Compare submission logs with checklist requirements.",
            "Relink or re-queue files not indexed by the portal.",
            "Share final checklist status and any remaining gaps.",
        ],
        expected_timeline="1-2 business days",
        escalation_triggers=["Visa or deadline risk", "Persistent mismatch after relink"],
        default_owner="Admissions Operations",
    ),
    "enrollment.admission_requirements": Playbook(
        title="Admission requirements guidance",
        short_answer="We provide the official eligibility checklist and required evidence for your profile.",
        required_documents=["target program", "prior degree summary"],
        process_steps=[
            "Confirm program and intake.",
            "Map applicant profile to official prerequisites.",
            "Share required documents and language/test conditions.",
            "Provide submission order and validation tips.",
        ],
        expected_timeline="Same day",
        escalation_triggers=["Profile does not match standard criteria", "Country-specific compliance concerns"],
        default_owner="Admissions Advising",
    ),
    "enrollment.offer_letter": Playbook(
        title="Offer letter issuance",
        short_answer="We confirm issuance readiness and provide the expected release date for your offer letter.",
        required_documents=["application ID", "passport name match confirmation"],
        process_steps=[
            "Validate decision status and completion of admission checks.",
            "Confirm identity and passport spelling consistency.",
            "Queue issuance or identify pending blockers.",
            "Provide issuance date and delivery channel.",
        ],
        expected_timeline="1-5 business days",
        escalation_triggers=["Embassy appointment within 48 hours", "Critical name mismatch"],
        default_owner="Admissions",
    ),
    "enrollment.identity_mismatch": Playbook(
        title="Identity mismatch resolution",
        short_answer="We align legal identity records across systems before enrollment or visa documentation is issued.",
        required_documents=["updated passport", "name change proof", "application ID"],
        process_steps=[
            "Collect legal name change evidence.",
            "Verify mismatch across portal, CRM, and admission records.",
            "Request backend identity correction.",
            "Reissue affected enrollment/offer documents.",
        ],
        expected_timeline="2-5 business days",
        escalation_triggers=["Upcoming visa/legal deadline", "Multiple systems still inconsistent"],
        default_owner="Registrar",
    ),
    "enrollment.duplicate_application": Playbook(
        title="Duplicate application merge",
        short_answer="We consolidate duplicate applications into one canonical applicant record.",
        required_documents=["both application references", "identity confirmation"],
        process_steps=[
            "Validate ownership of both application records.",
            "Pick canonical record and preserve audit trail.",
            "Merge documents, payment notes, and communication history.",
            "Confirm the single active application to the applicant.",
        ],
        expected_timeline="1-3 business days",
        escalation_triggers=["Different personal identifiers across duplicates", "Decision already issued on wrong record"],
        default_owner="Admissions Operations",
    ),
    "payments.invoice_request": Playbook(
        title="Invoice issuance",
        short_answer="We generate and send the tuition invoice with payment references and due dates.",
        required_documents=["student/application ID", "program and intake confirmation"],
        process_steps=[
            "Confirm billing entity and intake.",
            "Generate invoice from finance system.",
            "Validate amount, due date, and payment reference.",
            "Send invoice through requested channel.",
        ],
        expected_timeline="Same day to 1 business day",
        escalation_triggers=["Incorrect amount", "Invoice needed for visa in <24h"],
        default_owner="Finance",
    ),
    "payments.payment_confirmation": Playbook(
        title="Payment confirmation",
        short_answer="We trace payment settlement and issue confirmation once funds are matched.",
        required_documents=["transaction reference", "payment date", "payer name"],
        process_steps=[
            "Collect payment trace details.",
            "Match transaction in bank/PSP reconciliation.",
            "Resolve unmatched or pending settlements.",
            "Issue payment confirmation notice.",
        ],
        expected_timeline="1-3 business days",
        escalation_triggers=["No match after 3 business days", "Deadline at risk"],
        default_owner="Finance",
    ),
    "payments.installment_plan": Playbook(
        title="Installment plan request",
        short_answer="We assess eligibility and set up an installment schedule if policy conditions are met.",
        required_documents=["student/application ID", "financial justification (if required)"],
        process_steps=[
            "Confirm policy eligibility and account standing.",
            "Calculate installment schedule and due dates.",
            "Get applicant approval on repayment terms.",
            "Activate plan and issue payment instructions.",
        ],
        expected_timeline="2-4 business days",
        escalation_triggers=["Prior overdue balance", "Policy exception needed"],
        default_owner="Finance",
    ),
    "payments.refund_request": Playbook(
        title="Refund request handling",
        short_answer="We validate refund eligibility, compute refundable amount, and process payout.",
        required_documents=["payment proof", "withdrawal/change proof", "bank account details"],
        process_steps=[
            "Validate refund eligibility under policy.",
            "Compute refundable amount and deductions.",
            "Obtain finance approval.",
            "Initiate payout and send settlement confirmation.",
        ],
        expected_timeline="7-14 business days",
        escalation_triggers=["No payout after SLA", "Dispute over deductions"],
        default_owner="Finance",
    ),
    "payments.double_charge": Playbook(
        title="Double charge correction",
        short_answer="We verify duplicate charges and prioritize immediate correction/refund.",
        required_documents=["two charge references", "account statement excerpt"],
        process_steps=[
            "Validate duplicate transaction evidence.",
            "Freeze duplicate collection path.",
            "Authorize correction or immediate refund.",
            "Confirm resolved ledger balance in writing.",
        ],
        expected_timeline="Same day to 2 business days",
        escalation_triggers=["Critical financial hardship", "Repeated duplicate charging"],
        default_owner="Finance Escalations",
    ),
    "payments.fraud_alert": Playbook(
        title="Fraud alert response",
        short_answer="We verify suspicious links/messages and secure your payment account immediately.",
        required_documents=["suspicious link/message", "timestamp", "sender details"],
        process_steps=[
            "Classify the suspected fraud signal.",
            "Block affected payment channel if needed.",
            "Validate official payment instructions.",
            "Provide safe payment guidance and incident confirmation.",
        ],
        expected_timeline="Immediate to same day",
        escalation_triggers=["Credentials exposed", "Payment already executed to suspicious target"],
        default_owner="Information Security + Finance",
    ),
    "deadlines.application_deadline": Playbook(
        title="Application deadline clarification",
        short_answer="We confirm the official deadline and cut-off rules for your program intake.",
        required_documents=["program and intake"],
        process_steps=[
            "Check official deadline calendar.",
            "Confirm timezone and submission cut-off details.",
            "Share latest safe submission date.",
            "Suggest fallback intake if needed.",
        ],
        expected_timeline="Same day",
        escalation_triggers=["Portal outage near cut-off", "Conflicting published dates"],
        default_owner="Admissions",
    ),
    "deadlines.document_deadline": Playbook(
        title="Document deadline extension",
        short_answer="We review extension eligibility and confirm accepted interim evidence if approved.",
        required_documents=["reason for delay", "proof (courier/official notice)", "application ID"],
        process_steps=[
            "Validate reason and urgency.",
            "Check extension policy and constraints.",
            "Approve extension or define alternate evidence path.",
            "Share revised deadline and final checklist.",
        ],
        expected_timeline="1-2 business days",
        escalation_triggers=["Visa/legal dependency", "Extension policy exception needed"],
        default_owner="Admissions",
    ),
    "deadlines.payment_deadline": Playbook(
        title="Payment deadline impact",
        short_answer="We explain the impact of late payment and available remediation options.",
        required_documents=["invoice reference", "planned payment date"],
        process_steps=[
            "Confirm due date and payment policy state.",
            "Assess late-fee, hold, or enrollment risk.",
            "Propose immediate remediation plan.",
            "Confirm final safe settlement deadline.",
        ],
        expected_timeline="Same day",
        escalation_triggers=["Imminent deregistration risk", "System-applied penalties in error"],
        default_owner="Finance",
    ),
    "deadlines.medical_exception": Playbook(
        title="Medical exception review",
        short_answer="We process medical exception requests through formal review and documented evidence.",
        required_documents=["medical certificate", "timeline of impact", "application ID"],
        process_steps=[
            "Collect formal medical evidence.",
            "Validate exception eligibility criteria.",
            "Submit case to policy review board.",
            "Issue decision with adjusted timeline if approved.",
        ],
        expected_timeline="2-5 business days",
        escalation_triggers=["Critical enrollment loss risk", "Insufficient but urgent medical proof"],
        default_owner="Admissions Exceptions",
    ),
    "deadlines.visa_delay": Playbook(
        title="Visa delay support",
        short_answer="We evaluate late-arrival options and help secure enrollment continuity where possible.",
        required_documents=["embassy proof", "visa appointment details", "program/intake"],
        process_steps=[
            "Confirm visa delay evidence.",
            "Check late arrival policy and attendance constraints.",
            "Offer deferral, remote start, or bridge options.",
            "Document approved option and next deadline.",
        ],
        expected_timeline="1-3 business days",
        escalation_triggers=["No policy-fit option found", "Classes already in progress"],
        default_owner="International Office",
    ),
    "certificates.enrollment_certificate": Playbook(
        title="Enrollment certificate issuance",
        short_answer="We issue an enrollment certificate and prioritize urgent legal/permit cases.",
        required_documents=["student ID", "purpose of certificate"],
        process_steps=[
            "Validate enrollment status.",
            "Generate certificate with required fields.",
            "Apply signature/stamp workflow.",
            "Deliver PDF and, if needed, hard-copy instructions.",
        ],
        expected_timeline="Same day to 2 business days",
        escalation_triggers=["Embassy appointment within 24h", "Document authenticity dispute"],
        default_owner="Registrar",
    ),
    "certificates.transcript_request": Playbook(
        title="Transcript request",
        short_answer="We issue an official transcript with requested grading and credit details.",
        required_documents=["student ID", "transcript format target", "delivery destination"],
        process_steps=[
            "Confirm transcript scope and language.",
            "Extract validated academic records.",
            "Apply format rules (ECTS, grading scale).",
            "Issue secure transcript and send delivery confirmation.",
        ],
        expected_timeline="2-5 business days",
        escalation_triggers=["External transfer deadline", "Record mismatch detected"],
        default_owner="Registrar",
    ),
    "certificates.graduation_documents": Playbook(
        title="Graduation document availability",
        short_answer="We confirm issuance windows for graduation documents after final grade closure.",
        required_documents=["student ID", "graduation term"],
        process_steps=[
            "Verify final grade publication status.",
            "Check document production schedule.",
            "Share expected availability date.",
            "Provide collection/delivery instructions.",
        ],
        expected_timeline="3-10 business days after grade closure",
        escalation_triggers=["Employer/visa deadline", "Production delay beyond published window"],
        default_owner="Registrar",
    ),
    "certificates.urgent_embassy_letter": Playbook(
        title="Urgent embassy confirmation",
        short_answer="We prioritize urgent embassy letters with same-day verification where feasible.",
        required_documents=["embassy appointment proof", "passport ID", "student/application ID"],
        process_steps=[
            "Validate urgency and embassy appointment evidence.",
            "Verify enrollment/admission status.",
            "Issue stamped/signed confirmation letter.",
            "Deliver by fastest secure channel.",
        ],
        expected_timeline="Immediate to same day",
        escalation_triggers=["Appointment within 24h", "Signature authority unavailable"],
        default_owner="Registrar + Admissions",
    ),
    "certificates.grade_dispute": Playbook(
        title="Grade dispute process",
        short_answer="We register formal grade disputes and route them to academic review.",
        required_documents=["transcript section", "module code", "dispute rationale"],
        process_steps=[
            "Log formal dispute request.",
            "Collect supporting evidence and module context.",
            "Route case to academic review authority.",
            "Publish outcome and record updates.",
        ],
        expected_timeline="5-15 business days",
        escalation_triggers=["Graduation decision blocked", "Potential system grading error"],
        default_owner="Academic Affairs",
    ),
    "compliance.data_privacy_request": Playbook(
        title="Data privacy request (DSAR)",
        short_answer="We process data access/deletion requests under privacy policy and legal constraints.",
        required_documents=["identity proof", "request scope", "contact email"],
        process_steps=[
            "Verify requester identity.",
            "Classify access/deletion scope.",
            "Collect records across systems and legal holds.",
            "Deliver response and retention/deletion outcome.",
        ],
        expected_timeline="Up to 30 days",
        escalation_triggers=["Identity dispute", "Legal hold conflicts"],
        default_owner="Data Protection Office",
    ),
    "legal.complaint_escalation": Playbook(
        title="Formal complaint escalation",
        short_answer="We escalate unresolved complaints to management with full case context and deadlines.",
        required_documents=["prior complaint IDs", "timeline of unresolved actions"],
        process_steps=[
            "Validate complaint history and unresolved status.",
            "Assemble case file and prior responses.",
            "Escalate to management/legal review channel.",
            "Issue acknowledgement with decision timeline.",
        ],
        expected_timeline="1-3 business days for acknowledgement",
        escalation_triggers=["Regulatory/legal risk", "Repeated unresolved complaints"],
        default_owner="Management + Legal",
    ),
    "accessibility.disability_accommodation": Playbook(
        title="Disability accommodation",
        short_answer="We review disability accommodations and define approved support measures for assessments.",
        required_documents=["medical/disability documentation", "requested accommodation details"],
        process_steps=[
            "Verify submitted accommodation evidence.",
            "Review request with accessibility coordinator.",
            "Define approved support plan and exam adjustments.",
            "Notify faculty and student with implementation dates.",
        ],
        expected_timeline="3-7 business days",
        escalation_triggers=["Imminent exam date", "Insufficient evidence with high-impact risk"],
        default_owner="Accessibility Services",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def derive_priority(urgency: str, escalation_probability: float, edge_case: bool) -> str:
    if urgency == "critical" or escalation_probability >= 0.85:
        return "P1"
    if urgency == "high" or escalation_probability >= 0.65 or edge_case:
        return "P2"
    if urgency == "medium" or escalation_probability >= 0.35:
        return "P3"
    return "P4"


def build_standard_answer(playbook: Playbook, row: dict[str, Any]) -> str:
    program = str(row.get("program", "the selected program"))
    intake = str(row.get("intake", "the selected intake"))
    steps = " ".join(f"Step {idx + 1}: {step}" for idx, step in enumerate(playbook.process_steps))
    docs = ", ".join(playbook.required_documents) if playbook.required_documents else "No additional documents"
    return (
        f"For {program} ({intake}), follow this process. {playbook.short_answer} "
        f"Required documents: {docs}. Expected timeline: {playbook.expected_timeline}. {steps}"
    )


def readability_score(text: str, step_count: int) -> int:
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(word.strip(".,!?;:")) for word in words) / max(1, word_count)

    score = 100
    if word_count < 45:
        score -= 15
    if word_count > 180:
        score -= 10
    if avg_word_len > 7.5:
        score -= 8
    if step_count < 3:
        score -= 15
    if step_count > 6:
        score -= 8
    if "." not in text:
        score -= 10
    return max(0, min(100, score))


def transform_row(row: dict[str, Any], faq_index: int) -> dict[str, Any]:
    category = str(row["category"])
    subcategory = str(row["subcategory"])
    intent_key = f"{category}.{subcategory}"

    playbook = PLAYBOOKS.get(intent_key)
    if playbook is None:
        raise KeyError(f"Missing playbook for intent: {intent_key}")

    urgency = str(row["urgency"])
    escalation_probability = float(row["escalation_probability"])
    edge_case = bool(row["edge_case"])

    priority = derive_priority(urgency, escalation_probability, edge_case)
    escalation_required = edge_case or urgency in {"high", "critical"} or escalation_probability >= 0.70

    standardized_answer = build_standard_answer(playbook, row)
    score = readability_score(standardized_answer, len(playbook.process_steps))

    consistency_flags: list[str] = []
    if urgency == "critical" and priority != "P1":
        consistency_flags.append("critical_not_p1")
    if edge_case and not escalation_required:
        consistency_flags.append("edge_case_not_escalated")
    if len(playbook.process_steps) < 3:
        consistency_flags.append("insufficient_process_steps")

    return {
        "faq_id": f"FAQ-{faq_index:04d}",
        "source_inquiry_id": row["inquiry_id"],
        "intent_key": intent_key,
        "canonical_title": playbook.title,
        "question": row["inquiry_text"],
        "short_answer": playbook.short_answer,
        "standard_answer": standardized_answer,
        "required_documents": playbook.required_documents,
        "process_steps": playbook.process_steps,
        "expected_timeline": playbook.expected_timeline,
        "priority": priority,
        "escalation_required": escalation_required,
        "escalation_triggers": playbook.escalation_triggers,
        "owner_team": playbook.default_owner,
        "metadata": {
            "category": category,
            "subcategory": subcategory,
            "urgency": urgency,
            "escalation_probability": escalation_probability,
            "escalation_band": row["escalation_band"],
            "edge_case": edge_case,
            "language": row["language"],
            "channel": row["channel"],
            "program": row["program"],
            "intake": row["intake"],
        },
        "quality": {
            "ai_readability_score": score,
            "consistency_flags": consistency_flags,
            "consistency_passed": len(consistency_flags) == 0,
        },
    }


def build_faq_catalog(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["intent_key"])].append(entry)

    catalog: list[dict[str, Any]] = []
    for intent_key, group in sorted(grouped.items()):
        first = group[0]
        question_examples = [str(item["question"]) for item in group[:8]]
        catalog.append(
            {
                "intent_key": intent_key,
                "canonical_title": first["canonical_title"],
                "short_answer": first["short_answer"],
                "required_documents": first["required_documents"],
                "process_steps": first["process_steps"],
                "expected_timeline": first["expected_timeline"],
                "owner_team": first["owner_team"],
                "escalation_triggers": first["escalation_triggers"],
                "sample_questions": question_examples,
                "entry_count": len(group),
            }
        )

    return catalog


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(entries: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    priorities: dict[str, int] = defaultdict(int)
    owners: dict[str, int] = defaultdict(int)
    failed_consistency = 0
    readability_scores: list[int] = []

    for entry in entries:
        priorities[str(entry["priority"])] += 1
        owners[str(entry["owner_team"])] += 1
        quality = entry["quality"]
        readability_scores.append(int(quality["ai_readability_score"]))
        if not bool(quality["consistency_passed"]):
            failed_consistency += 1

    avg_readability = round(sum(readability_scores) / max(1, len(readability_scores)), 2)

    return {
        "total_entries": len(entries),
        "total_intents": len(catalog),
        "consistency_failures": failed_consistency,
        "consistency_pass_rate": round((len(entries) - failed_consistency) / max(1, len(entries)), 4),
        "avg_ai_readability_score": avg_readability,
        "priority_distribution": dict(sorted(priorities.items())),
        "owner_distribution": dict(sorted(owners.items())),
        "schema": {
            "entry_file": "synthetic_faq_kb.jsonl",
            "catalog_file": "synthetic_faq_catalog.json",
            "summary_file": "synthetic_faq_summary.json",
        },
    }


def build_process_markdown() -> str:
    return """# Synthetic FAQ Knowledge Base Build Process

## Goal
Build a machine-readable, logically consistent FAQ system from simulated student inquiries.

## Input
- `test/fixtures/synthetic_inquiries.jsonl`
- Each row contains intent metadata (`category`, `subcategory`), urgency, escalation probability, and inquiry text.

## Output
- `test/fixtures/synthetic_faq_kb.jsonl`: normalized FAQ entries (one per inquiry)
- `test/fixtures/synthetic_faq_catalog.json`: scalable intent-level FAQ catalog
- `test/fixtures/synthetic_faq_summary.json`: quality and consistency metrics

## Standardization Pipeline
1. Load each synthetic inquiry record.
2. Map `category.subcategory` to a canonical playbook.
3. Build a standardized answer with:
   - concise short answer
   - required documents
   - explicit step-by-step process
   - expected timeline
4. Derive machine-usable routing fields:
   - priority (`P1`-`P4`)
   - owner team
   - escalation required (boolean)
   - escalation triggers
5. Run logical consistency checks:
   - `critical` urgency must map to `P1`
   - edge-case scenarios must be escalatable
   - process must include at least 3 steps
6. Compute AI-readability score for downstream LLM retrieval quality control.

## Consistency Rules
- Intent keys must have a defined playbook.
- Every FAQ entry must include process steps, owner team, and timeline.
- Escalation policy must align with urgency and risk probability.

## Scalability Characteristics
- Intent-centric architecture (`intent_key`) supports easy extension.
- New scenarios can be added by extending playbooks only.
- Catalog + entry split supports retrieval by:
  - intent-level FAQ lookup
  - case-level response generation

## Usage
```bash
python scripts/dev/build_synthetic_faq_kb.py
```
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fixtures_dir = root / "test" / "fixtures"

    input_path = fixtures_dir / "synthetic_inquiries.jsonl"
    output_entries_path = fixtures_dir / "synthetic_faq_kb.jsonl"
    output_catalog_path = fixtures_dir / "synthetic_faq_catalog.json"
    output_summary_path = fixtures_dir / "synthetic_faq_summary.json"
    process_doc_path = fixtures_dir / "synthetic_faq_process.md"

    source_rows = load_jsonl(input_path)
    entries = [transform_row(row, idx) for idx, row in enumerate(source_rows, start=1)]
    catalog = build_faq_catalog(entries)
    summary = build_summary(entries, catalog)

    save_jsonl(output_entries_path, entries)
    save_json(output_catalog_path, catalog)
    save_json(output_summary_path, summary)
    process_doc_path.write_text(build_process_markdown(), encoding="utf-8")

    print(f"Built FAQ entries: {len(entries)}")
    print(f"Built FAQ intents: {len(catalog)}")
    print(f"Consistency failures: {summary['consistency_failures']}")
    print(f"Avg readability: {summary['avg_ai_readability_score']}")
    print(f"Saved: {output_entries_path}")
    print(f"Saved: {output_catalog_path}")
    print(f"Saved: {output_summary_path}")
    print(f"Saved: {process_doc_path}")


if __name__ == "__main__":
    main()

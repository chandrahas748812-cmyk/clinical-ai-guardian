"""Tests: PII detection and redaction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.privacy.detector import PIIDetector
from src.privacy.redactor import PIIRedactor


SAMPLE = (
    "Patient: James Carter\nDOB: 03/14/1968\nMRN: MC-884210\n"
    "SSN: 123-45-6789\nPhone: (214) 555-0147\n"
    "Email: j.carter.fake@example.com\nDr. Emily Nguyen noted diabetes."
)


def test_detect_ssn():
    dets = PIIDetector().detect("SSN: 123-45-6789")
    assert any(d.entity_type == "SSN" and d.text == "123-45-6789" for d in dets)


def test_detect_phone_email_mrn():
    dets = PIIDetector().detect(SAMPLE)
    types = {d.entity_type for d in dets}
    assert {"PHONE", "EMAIL", "MRN", "DATE"} <= types


def test_detect_person_name():
    dets = PIIDetector().detect("Patient: James Carter was seen by Dr. Emily Nguyen.")
    persons = [d.text for d in dets if d.entity_type == "PERSON"]
    assert "James Carter" in persons
    assert "Emily Nguyen" in persons


def test_threshold_filters_low_confidence():
    det = PIIDetector(config={"thresholds": {"SSN": 0.9999},
                              "default_threshold": 0.99})
    # space-separated SSN has confidence 0.90 < 0.9999 -> filtered
    dets = det.detect("SSN 123 45 6789")
    assert not any(d.entity_type == "SSN" for d in dets)


def test_redact_replaces_with_placeholders():
    res = PIIRedactor().redact(SAMPLE)
    assert "123-45-6789" not in res.redacted_text
    assert "j.carter.fake@example.com" not in res.redacted_text
    assert "[SSN_1]" in res.redacted_text
    assert "[EMAIL_1]" in res.redacted_text
    assert res.entities_redacted["SSN"] >= 1


def test_redaction_map_reversible():
    redactor = PIIRedactor()
    res = redactor.redact(SAMPLE)
    restored = res.redaction_map.restore(res.redacted_text)
    assert restored == SAMPLE


def test_seal_hash_does_not_contain_pii():
    res = PIIRedactor().redact(SAMPLE)
    seal = res.redaction_map.seal_hash()
    assert "123-45-6789" not in seal
    assert len(seal) == 64  # sha256 hex


def test_scan_for_leak_detects_raw_pii():
    redactor = PIIRedactor()
    res = redactor.redact(SAMPLE)
    leaked = redactor.scan_for_leak("The SSN is 123-45-6789 ok?", res.redaction_map)
    assert "SSN" in leaked
    clean = redactor.scan_for_leak(res.redacted_text, res.redaction_map)
    assert clean == []


def test_repeated_entity_reuses_placeholder():
    res = PIIRedactor().redact("Call (214) 555-0147 or (214) 555-0147 today.")
    assert res.redacted_text.count("[PHONE_1]") == 2

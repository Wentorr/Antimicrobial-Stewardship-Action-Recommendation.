from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260707
TRAIN_PER_TOP_ACTION = 440
TEST_PER_TOP_ACTION = 134

ACTIONS = [
    "deescalate_to_narrow_beta_lactam",
    "escalate_to_carbapenem",
    "switch_to_oral_stepdown",
    "stop_antibiotics_monitor",
    "add_mrsa_coverage",
    "remove_mrsa_coverage",
    "dose_adjust_for_renal",
    "request_source_control",
    "repeat_cultures_and_review",
    "request_id_consult",
    "add_anaerobic_coverage",
    "flag_allergy_review",
]

CARE_UNITS = ["ward", "stepdown", "icu", "ed_observation", "oncology"]
SYNDROMES = [
    "urinary_tract",
    "pneumonia",
    "intra_abdominal",
    "skin_soft_tissue",
    "bloodstream",
    "catheter_related",
    "biliary",
    "bone_joint",
]
CULTURE_SITES = ["urine", "sputum", "blood", "wound", "line_tip", "bile", "peritoneal_fluid"]
ORGANISMS = [
    "ecoli",
    "klebsiella",
    "pseudomonas",
    "enterococcus",
    "staph_aureus",
    "coagulase_negative_staph",
    "bacteroides",
    "mixed_flora",
    "no_growth",
]
RESISTANCE = ["none", "ESBL", "AmpC", "MRSA", "VRE", "carbapenemase", "inducible_clinda", "pending"]
REGIMENS = [
    "ceftriaxone",
    "cefepime",
    "piperacillin_tazobactam",
    "meropenem",
    "vancomycin",
    "vancomycin_plus_cefepime",
    "ceftriaxone_plus_metronidazole",
    "ciprofloxacin",
    "linezolid",
    "ampicillin",
    "daptomycin",
]
RENAL = ["normal", "mild_impairment", "egfr_30_60", "egfr_lt_30", "aki"]
ALLERGY = ["none", "rash_history", "severe_beta_lactam", "unclear_anaphylaxis"]
FEVER = ["resolved", "improving", "persistent", "new_spike"]
WBC = ["normalizing", "stable_high", "rising", "falling_low"]
STATUS = ["stable", "borderline", "unstable"]
SOURCE = ["controlled", "uncertain", "drain_pending", "line_not_removed", "obstructed"]
ORAL = ["not_ready", "maybe_ready", "ready"]
MRSA = ["negative", "positive", "not_done"]
ESBL_RATE = ["low", "medium", "high"]
HOSP = ["no_recent", "recent_30d", "long_term_care", "dialysis_exposure"]


def _choice(rng: random.Random, values: list[str], weights: list[float] | None = None) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _has_beta_lactam(regimen: str) -> bool:
    return any(key in regimen for key in ["cef", "piperacillin", "meropenem", "ampicillin"])


def _broad_regimen(regimen: str) -> bool:
    return regimen in {"cefepime", "piperacillin_tazobactam", "meropenem", "vancomycin_plus_cefepime"}


def _has_mrsa_coverage(regimen: str) -> bool:
    return any(key in regimen for key in ["vancomycin", "linezolid", "daptomycin"])


def _gram_negative(organism: str) -> bool:
    return organism in {"ecoli", "klebsiella", "pseudomonas"}


def _note(row: dict[str, object]) -> str:
    resistance_phrase = {
        "none": "susceptibility panel shows no named resistance marker",
        "ESBL": "lab comment flags an ESBL phenotype",
        "AmpC": "screen suggests possible AmpC induction risk",
        "MRSA": "culture marker is consistent with MRSA",
        "VRE": "enterococcal isolate carries a vancomycin resistance marker",
        "carbapenemase": "molecular panel flags a carbapenemase marker",
        "inducible_clinda": "D-test style note suggests inducible clindamycin resistance",
        "pending": "final susceptibility data are still pending",
    }[str(row["resistance_marker"])]

    return (
        f"{row['care_unit']} stewardship packet for {row['infection_syndrome']} infection: "
        f"{row['culture_site']} culture reports {row['organism_group']}; {resistance_phrase}. "
        f"Current regimen is {row['current_regimen']} on day {row['days_on_therapy']}. "
        f"Renal status is {row['renal_band']}; allergy history is {row['allergy_flag']}. "
        f"Fever is {row['fever_trend']} and WBC trend is {row['wbc_trend']} with hemodynamic status "
        f"{row['hemodynamic_status']}. Source control is {row['source_control_status']}. "
        f"Oral route readiness is {row['oral_route_ready']}; MRSA screen is {row['mrsa_screen']}. "
        f"Local ESBL rate is {row['local_esbl_rate_band']} and exposure history is {row['recent_healthcare_exposure']}."
    )


def _score_actions(row: dict[str, object], rng: random.Random) -> dict[str, float]:
    scores = {action: 0.05 + rng.random() * 0.02 for action in ACTIONS}

    syndrome = str(row["infection_syndrome"])
    organism = str(row["organism_group"])
    marker = str(row["resistance_marker"])
    regimen = str(row["current_regimen"])
    renal = str(row["renal_band"])
    allergy = str(row["allergy_flag"])
    fever = str(row["fever_trend"])
    wbc = str(row["wbc_trend"])
    status = str(row["hemodynamic_status"])
    source = str(row["source_control_status"])
    oral = str(row["oral_route_ready"])
    mrsa = str(row["mrsa_screen"])
    esbl = str(row["local_esbl_rate_band"])
    exposure = str(row["recent_healthcare_exposure"])
    day = int(row["days_on_therapy"])

    if source in {"drain_pending", "line_not_removed", "obstructed"}:
        scores["request_source_control"] += 2.8
        if status != "stable":
            scores["request_source_control"] += 1.0
        if syndrome in {"intra_abdominal", "catheter_related", "biliary"}:
            scores["request_source_control"] += 0.8

    if renal in {"egfr_lt_30", "aki"}:
        scores["dose_adjust_for_renal"] += 2.0
        if _has_mrsa_coverage(regimen) or regimen in {"piperacillin_tazobactam", "meropenem", "cefepime"}:
            scores["dose_adjust_for_renal"] += 1.4
        if status == "unstable":
            scores["dose_adjust_for_renal"] += 0.4

    if allergy in {"severe_beta_lactam", "unclear_anaphylaxis"} and _has_beta_lactam(regimen):
        scores["flag_allergy_review"] += 3.1
        if marker in {"ESBL", "AmpC", "carbapenemase"}:
            scores["flag_allergy_review"] += 0.5

    if marker in {"ESBL", "AmpC"} and _gram_negative(organism):
        scores["escalate_to_carbapenem"] += 2.2
        if esbl == "high" or exposure in {"recent_30d", "long_term_care", "dialysis_exposure"}:
            scores["escalate_to_carbapenem"] += 1.0
        if regimen != "meropenem":
            scores["escalate_to_carbapenem"] += 0.8

    if marker == "carbapenemase":
        scores["request_id_consult"] += 3.4
        scores["escalate_to_carbapenem"] += 0.8

    if marker == "VRE" and organism == "enterococcus":
        scores["request_id_consult"] += 1.6
        if regimen in {"vancomycin", "ampicillin"}:
            scores["request_id_consult"] += 1.3

    if marker == "MRSA" or (mrsa == "positive" and syndrome in {"pneumonia", "skin_soft_tissue", "bloodstream"}):
        if not _has_mrsa_coverage(regimen):
            scores["add_mrsa_coverage"] += 3.0
        else:
            scores["remove_mrsa_coverage"] += 0.25

    if mrsa == "negative" and _has_mrsa_coverage(regimen) and marker != "MRSA":
        scores["remove_mrsa_coverage"] += 2.4
        if status == "stable" and fever in {"resolved", "improving"}:
            scores["remove_mrsa_coverage"] += 0.8

    if syndrome in {"intra_abdominal", "biliary"} and "metronidazole" not in regimen and organism in {"bacteroides", "mixed_flora"}:
        scores["add_anaerobic_coverage"] += 3.0
        if source != "controlled":
            scores["add_anaerobic_coverage"] += 0.8

    if organism == "no_growth" and day >= 3 and status == "stable" and fever in {"resolved", "improving"}:
        scores["stop_antibiotics_monitor"] += 3.2
        if wbc in {"normalizing", "falling_low"}:
            scores["stop_antibiotics_monitor"] += 0.7

    if oral == "ready" and day >= 3 and status == "stable" and marker not in {"carbapenemase", "pending"}:
        scores["switch_to_oral_stepdown"] += 2.8
        if fever in {"resolved", "improving"}:
            scores["switch_to_oral_stepdown"] += 0.7

    if _broad_regimen(regimen) and marker in {"none", "pending"} and organism in {"ecoli", "klebsiella", "staph_aureus"}:
        scores["deescalate_to_narrow_beta_lactam"] += 2.5
        if status == "stable":
            scores["deescalate_to_narrow_beta_lactam"] += 0.9
        if day >= 2:
            scores["deescalate_to_narrow_beta_lactam"] += 0.4

    if marker == "pending" or (wbc == "rising" and fever in {"persistent", "new_spike"} and source == "uncertain"):
        scores["repeat_cultures_and_review"] += 2.6
        if status != "stable":
            scores["repeat_cultures_and_review"] += 0.6

    # Visible calibration cues make some rows non-obvious without making them hidden.
    lot_primary = str(row["lot_primary_bias"])
    lot_secondary = str(row["lot_secondary_bias"])
    lot_suppressed = str(row["lot_suppressed_bias"])
    strength = float(row["calibration_strength"])
    for action, family in {
        "request_source_control": "source",
        "repeat_cultures_and_review": "diagnostic",
        "request_id_consult": "consult",
        "dose_adjust_for_renal": "safety",
        "flag_allergy_review": "safety",
        "add_mrsa_coverage": "coverage",
        "remove_mrsa_coverage": "deescalation",
        "deescalate_to_narrow_beta_lactam": "deescalation",
        "switch_to_oral_stepdown": "deescalation",
        "escalate_to_carbapenem": "coverage",
        "add_anaerobic_coverage": "coverage",
        "stop_antibiotics_monitor": "deescalation",
    }.items():
        if family == lot_primary:
            scores[action] += 0.85 * strength
        if family == lot_secondary:
            scores[action] += 0.35 * strength
        if family == lot_suppressed:
            scores[action] -= 0.65 * strength

    return scores


def _make_row(idx: int, rng: random.Random, hard_bias: bool) -> dict[str, object]:
    syndrome = _choice(rng, SYNDROMES, [1.3, 1.1, 1.0, 1.0, 0.9, 0.75, 0.7, 0.55])
    organism_weights = [1.4, 1.1, 0.85, 0.75, 0.95, 0.4, 0.5, 0.45, 0.8]
    marker_weights = [2.4, 0.8, 0.6, 0.55, 0.35, 0.25, 0.25, 0.9]
    if hard_bias:
        marker_weights = [1.3, 1.0, 0.85, 0.75, 0.5, 0.55, 0.35, 1.0]
    row = {
        "case_id": f"ams_raw_{idx:05d}",
        "age_band": _choice(rng, ["18_39", "40_64", "65_79", "80_plus"], [0.18, 0.38, 0.29, 0.15]),
        "care_unit": _choice(rng, CARE_UNITS, [0.42, 0.18, 0.16, 0.12, 0.12]),
        "infection_syndrome": syndrome,
        "culture_site": _choice(rng, CULTURE_SITES),
        "organism_group": _choice(rng, ORGANISMS, organism_weights),
        "resistance_marker": _choice(rng, RESISTANCE, marker_weights),
        "current_regimen": _choice(rng, REGIMENS),
        "days_on_therapy": rng.randint(1, 7),
        "renal_band": _choice(rng, RENAL, [0.46, 0.18, 0.17, 0.11, 0.08]),
        "allergy_flag": _choice(rng, ALLERGY, [0.70, 0.14, 0.08, 0.08]),
        "fever_trend": _choice(rng, FEVER, [0.26, 0.30, 0.28, 0.16]),
        "wbc_trend": _choice(rng, WBC, [0.33, 0.27, 0.27, 0.13]),
        "hemodynamic_status": _choice(rng, STATUS, [0.68, 0.22, 0.10]),
        "source_control_status": _choice(rng, SOURCE, [0.48, 0.20, 0.12, 0.10, 0.10]),
        "oral_route_ready": _choice(rng, ORAL, [0.48, 0.24, 0.28]),
        "mrsa_screen": _choice(rng, MRSA, [0.50, 0.20, 0.30]),
        "local_esbl_rate_band": _choice(rng, ESBL_RATE, [0.38, 0.35, 0.27]),
        "recent_healthcare_exposure": _choice(rng, HOSP, [0.50, 0.24, 0.15, 0.11]),
        "calibration_lot": _choice(rng, [f"lot_{i:02d}" for i in range(12)]),
        "lot_primary_bias": _choice(rng, ["coverage", "deescalation", "source", "safety", "diagnostic", "consult"]),
        "lot_secondary_bias": _choice(rng, ["coverage", "deescalation", "source", "safety", "diagnostic", "consult"]),
        "lot_suppressed_bias": _choice(rng, ["coverage", "deescalation", "source", "safety", "diagnostic", "consult"]),
        "calibration_strength": round(rng.uniform(0.2, 1.0 if hard_bias else 0.75), 3),
    }

    # Ensure some clinically coherent pairings without making the row deterministic from one column.
    if row["resistance_marker"] == "MRSA":
        row["organism_group"] = "staph_aureus"
        row["culture_site"] = _choice(rng, ["blood", "wound", "sputum"])
    if row["resistance_marker"] == "VRE":
        row["organism_group"] = "enterococcus"
    if row["organism_group"] == "no_growth":
        row["resistance_marker"] = _choice(rng, ["none", "pending"], [0.7, 0.3])
    if syndrome == "catheter_related":
        row["culture_site"] = _choice(rng, ["blood", "line_tip"], [0.65, 0.35])
    if syndrome in {"intra_abdominal", "biliary"}:
        row["culture_site"] = _choice(rng, ["peritoneal_fluid", "bile", "blood"], [0.5, 0.35, 0.15])
    if hard_bias and rng.random() < 0.45:
        row["source_control_status"] = _choice(rng, ["uncertain", "drain_pending", "line_not_removed", "obstructed"])
        row["oral_route_ready"] = _choice(rng, ["maybe_ready", "ready", "not_ready"], [0.45, 0.25, 0.30])

    row["stewardship_note"] = _note(row)
    candidate_actions = ACTIONS.copy()
    rng.shuffle(candidate_actions)
    row["candidate_actions"] = "|".join(candidate_actions)

    scores = _score_actions(row, rng)
    ranked = sorted(ACTIONS, key=lambda action: (-scores[action], action))
    row["action_manifest"] = json.dumps({"ranked_actions": ranked[:3]}, separators=(",", ":"))
    row["action_scores_json"] = json.dumps({action: round(scores[action], 5) for action in ACTIONS}, separators=(",", ":"))
    top = ranked[0]
    family_map = {
        "request_source_control": "source",
        "repeat_cultures_and_review": "diagnostic",
        "request_id_consult": "consult",
        "dose_adjust_for_renal": "safety",
        "flag_allergy_review": "safety",
        "add_mrsa_coverage": "coverage",
        "remove_mrsa_coverage": "deescalation",
        "deescalate_to_narrow_beta_lactam": "deescalation",
        "switch_to_oral_stepdown": "deescalation",
        "escalate_to_carbapenem": "coverage",
        "add_anaerobic_coverage": "coverage",
        "stop_antibiotics_monitor": "deescalation",
    }
    row["top_action_family"] = family_map[top]
    gap = scores[ranked[0]] - scores[ranked[1]]
    row["difficulty_tier"] = "hard" if gap < 0.55 or hard_bias else ("medium" if gap < 1.2 else "easy")
    row["split_bucket"] = "test" if hard_bias else "train"
    return row


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    train_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    idx = 0

    while (
        any(train_counts[action] < TRAIN_PER_TOP_ACTION for action in ACTIONS)
        or any(test_counts[action] < TEST_PER_TOP_ACTION for action in ACTIONS)
    ):
        hard_bias = rng.random() < 0.45
        row = _make_row(idx, rng, hard_bias)
        top_action = json.loads(str(row["action_manifest"]))["ranked_actions"][0]
        if test_counts[top_action] < TEST_PER_TOP_ACTION and (row["difficulty_tier"] in {"medium", "hard"} or hard_bias):
            row["split_bucket"] = "test"
            test_counts[top_action] += 1
            rows.append(row)
        elif train_counts[top_action] < TRAIN_PER_TOP_ACTION:
            row["split_bucket"] = "train"
            train_counts[top_action] += 1
            rows.append(row)
        idx += 1
        if idx > 120000:
            raise RuntimeError("could not build balanced action dataset")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out = Path(__file__).resolve().parent / "data.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print("train top-action counts:", dict(train_counts))
    print("test top-action counts:", dict(test_counts))


if __name__ == "__main__":
    main()

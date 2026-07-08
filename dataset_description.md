# Antimicrobial Stewardship Note Action Sequencing Dataset

## Overview

This dataset contains de-identified synthetic antimicrobial stewardship packets for hospital culture follow-up. Each row represents a single stewardship review case with patient context, culture findings, resistance-marker information, current antimicrobial therapy, renal and allergy safety fields, source-control status, oral stepdown readiness, local antibiogram context, and a short natural-language note.

The benchmark supports NLP systems that convert stewardship notes and normalized clinical context into a three-action JSON sequence. The target is a top-three action manifest ordered from highest to lower action priority.

The dataset is generated locally by the included `generate_raw.py` script using deterministic stewardship templates and a fixed random seed. It does not contain real patient data, hospital names, medical record numbers, prescriber names, or external clinical source labels.

## File Structure

The raw upload contains exactly these top-level files:

| File | Description |
|---|---|
| `data.csv` | Complete generated stewardship cases, public fields, target action manifests, and private scoring metadata used by `prepare.py`. |
| `generate_raw.py` | Deterministic Python script that creates `data.csv` from a fixed seed. |

The Eris prepare script creates these challenge files:

| Prepared File | Description |
|---|---|
| `public/train.csv` | Public training rows with stewardship fields and target `action_manifest`. |
| `public/test.csv` | Public held-out rows with stewardship fields; target manifests are removed. |
| `public/sample_submission.csv` | Random valid submission template with `case_id` and `action_manifest`. |
| `private/answers.csv` | Private targets, hidden action relevance scores, and robustness groups derived by `prepare.py` for grading. |

## Raw Columns

| Column | Type | Description |
|---|---|---|
| `case_id` | string | Raw case identifier before hashing. |
| `age_band` | categorical | Patient age bucket. |
| `care_unit` | categorical | Ward, stepdown, ICU, ED observation, or oncology context. |
| `infection_syndrome` | categorical | Primary infection syndrome. |
| `culture_site` | categorical | Source of the culture report. |
| `organism_group` | categorical | Reported organism group or no-growth state. |
| `resistance_marker` | categorical | Named resistance or susceptibility clue. |
| `current_regimen` | categorical | Current antimicrobial regimen at review time. |
| `days_on_therapy` | integer | Number of therapy days before review. |
| `renal_band` | categorical | Renal function and kidney-injury bucket. |
| `allergy_flag` | categorical | Allergy history bucket. |
| `fever_trend` | categorical | Fever trajectory. |
| `wbc_trend` | categorical | White-blood-cell trend. |
| `hemodynamic_status` | categorical | Stability state. |
| `source_control_status` | categorical | Source-control state such as controlled, uncertain, drain pending, line not removed, or obstructed. |
| `oral_route_ready` | categorical | Oral stepdown readiness. |
| `mrsa_screen` | categorical | MRSA screening state. |
| `local_esbl_rate_band` | categorical | Local ESBL prevalence bucket. |
| `recent_healthcare_exposure` | categorical | Recent healthcare exposure pattern. |
| `calibration_lot` | categorical | Local policy lot code. |
| `lot_primary_bias` | categorical | Action family with strongest visible local policy emphasis. |
| `lot_secondary_bias` | categorical | Action family with weaker visible local policy emphasis. |
| `lot_suppressed_bias` | categorical | Action family visibly down-weighted by local policy. |
| `calibration_strength` | float | Magnitude of the visible local policy adjustment. |
| `candidate_actions` | string | Pipe-separated list of valid actions. |
| `stewardship_note` | string | Natural-language summary of the stewardship packet. |
| `action_manifest` | JSON string | Target top-three action sequence. |
| `action_scores_json` | JSON string | Private action relevance scores used for NDCG-style grading. |
| `top_action_family` | categorical | Private family of the highest-relevance action. |
| `difficulty_tier` | categorical | Easy, medium, or hard based on action-score separation and competing cues. |
| `split_bucket` | categorical | Internal train/test split marker consumed by `prepare.py`. |

`prepare.py` derives two additional private robustness columns for `private/answers.csv`: `resistance_safety_group`, which combines difficulty tier, top action family, resistance marker, and renal band; and `syndrome_action_group`, which combines syndrome, care unit, and top action family.

## Public Prepared Columns

| Column | Type | Present In | Description |
|---|---|---|---|
| `case_id` | string | train, test, submission | Hashed case identifier. |
| `age_band` | categorical | train, test | Patient age bucket. |
| `care_unit` | categorical | train, test | Clinical review location. |
| `infection_syndrome` | categorical | train, test | Primary infection syndrome. |
| `culture_site` | categorical | train, test | Culture source. |
| `organism_group` | categorical | train, test | Organism group. |
| `resistance_marker` | categorical | train, test | Resistance marker. |
| `current_regimen` | categorical | train, test | Current antimicrobial regimen. |
| `days_on_therapy` | integer | train, test | Therapy day at review. |
| `renal_band` | categorical | train, test | Renal safety bucket. |
| `allergy_flag` | categorical | train, test | Allergy history bucket. |
| `fever_trend` | categorical | train, test | Fever trajectory. |
| `wbc_trend` | categorical | train, test | WBC trajectory. |
| `hemodynamic_status` | categorical | train, test | Patient stability bucket. |
| `source_control_status` | categorical | train, test | Source-control state. |
| `oral_route_ready` | categorical | train, test | Oral stepdown readiness. |
| `mrsa_screen` | categorical | train, test | MRSA screen state. |
| `local_esbl_rate_band` | categorical | train, test | Local ESBL prevalence bucket. |
| `recent_healthcare_exposure` | categorical | train, test | Recent healthcare exposure. |
| `calibration_lot` | categorical | train, test | Local policy lot. |
| `lot_primary_bias` | categorical | train, test | Primary policy emphasis family. |
| `lot_secondary_bias` | categorical | train, test | Secondary policy emphasis family. |
| `lot_suppressed_bias` | categorical | train, test | Suppressed policy family. |
| `calibration_strength` | float | train, test | Policy adjustment strength. |
| `candidate_actions` | string | train, test | Pipe-separated action catalog for the row. |
| `stewardship_note` | string | train, test | Natural-language stewardship summary. |
| `action_manifest` | JSON string | train, submission | Ranked top-three action sequence. |

## Data Characteristics

- Training rows: 5,200.
- Test rows: 1,600.
- Candidate actions per case: 12.
- Output actions per case: 3 ranked actions.
- Hard held-out cases emphasize competing stewardship cues, rare resistance markers, renal/allergy safety conflicts, and local policy calibration shifts.
- Random valid top-three action lists are expected to score near 0.10 to 0.25 because the action catalog is large and the grader checks the strictest hidden worst-group robustness score.

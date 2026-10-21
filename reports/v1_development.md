# KASPARRO Claim-Verification: Development Evaluation (V1)

**Run Status**: COMPLETED (40/40 Answers)  
**Date**: 2026-10-21T11:15:46.927401+00:00  
**Provider**: `benchmark-simulator` | **Model**: `benchmark-engine-v1`  
**Prompt Version**: `v1` (SHA-256: `d988649f0beddba3844fa08fcc5d0938ab281a4f9acb016e419eb9a331de4c3a`)  
**Fact Version**: `speaker-catalogue-v4`  
**Total Run Duration**: 0.16s  

---

## 1. Deterministic Development Metrics

| Metric | Value | Definition |
| :--- | :--- | :--- |
| **Gold Claims (G)** | 376 | Total ground truth claims across scheduled answers |
| **Accepted Predictions (P)** | 374 | Total valid extracted claims |
| **Exact Span Matches (M)** | 370 | Claims matching exact `(answer_id, start_char, end_char)` |
| **Correct Label Matches (C)** | 356 | Exact span matches with correct verification label |
| **Extraction Precision** | 0.9893 | M / P |
| **Extraction Recall** | 0.9840 | M / G |
| **Extraction F1** | 0.9867 | 2*M / (P + G) |
| **Joint Precision** | 0.9519 | C / P |
| **Joint Recall** | 0.9468 | C / G |
| **Joint F1** | 0.9493 | 2*C / (P + G) |
| **Gold Claim Coverage** | 0.9468 | C / G |
| **Matched Label Accuracy** | 0.9622 | C / M (Diagnostic only, not substituted for Joint F1) |

---

## 2. Operational Statistics & Error Summary

- **Total Scheduled Answers**: 40
- **Successfully Evaluated Answers**: 40
- **Failed Answers**: 0
- **Missed Gold Claims**: 6
- **Extra Unmatched Predictions**: 4
- **Label Mismatches (Wrong Labels)**: 14

### Observed Label Mismatches:
| Claim ID | Gold Label | Predicted Label | Quote Snippet |
| :--- | :--- | :--- | :--- |
| `A01:1440:1619` | **contradicted** | **supported** | The stated laboratory playback time for ClearTrail Mini, using a new batter... |
| `A04:2363:2540` | **contradicted** | **unknown** | In the 2025 India catalogue, before the current-year revision, StoneWave Ma... |
| `A06:495:697` | **unknown** | **supported** | No version of ClearTrail Plus has ever been offered in blue, including prev... |
| `A06:1622:1801` | **contradicted** | **supported** | The stated laboratory playback time for ClearTrail Plus, using a new batter... |
| `A07:306:511` | **unknown** | **supported** | A retail purchase of ClearTrail Plus in Germany qualifies for the same 12-m... |
| `A12:335:515` | **unknown** | **supported** | The official India catalogue price of StoneWave Go will drop by INR 300 dur... |
| `A14:2019:2198` | **contradicted** | **supported** | The stated laboratory playback time for ClearTrail Plus, using a new batter... |
| `A19:371:555` | **contradicted** | **supported** | For an authorized India purchase with the original invoice and a manufactur... |

### Missed Gold Claims by Difficulty Tag:
| Difficulty Tag | Missed Count |
| :--- | :--- |
| `model_scope` | 2 |
| `explicit_false` | 2 |
| `outdated_fact` | 1 |
| `negation` | 1 |
| `scope` | 1 |
| `comparison` | 1 |
| `arithmetic` | 1 |
| `qualification` | 1 |
| `unit_conversion` | 1 |


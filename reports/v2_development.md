# KASPARRO Claim-Verification: Development Evaluation (V2)

**Run Status**: COMPLETED (40/40 Answers)  
**Date**: 2026-10-21T11:15:47.423077+00:00  
**Provider**: `benchmark-simulator` | **Model**: `benchmark-engine-v2`  
**Prompt Version**: `v2` (SHA-256: `54586b58234d66a0682b8ed397159f0bb1e18555160cc2105bc439f2d1c0c89c`)  
**Fact Version**: `speaker-catalogue-v4`  
**Total Run Duration**: 0.18s  

---

## 1. Deterministic Development Metrics

| Metric | Value | Definition |
| :--- | :--- | :--- |
| **Gold Claims (G)** | 376 | Total ground truth claims across scheduled answers |
| **Accepted Predictions (P)** | 374 | Total valid extracted claims |
| **Exact Span Matches (M)** | 370 | Claims matching exact `(answer_id, start_char, end_char)` |
| **Correct Label Matches (C)** | 362 | Exact span matches with correct verification label |
| **Extraction Precision** | 0.9893 | M / P |
| **Extraction Recall** | 0.9840 | M / G |
| **Extraction F1** | 0.9867 | 2*M / (P + G) |
| **Joint Precision** | 0.9679 | C / P |
| **Joint Recall** | 0.9628 | C / G |
| **Joint F1** | 0.9653 | 2*C / (P + G) |
| **Gold Claim Coverage** | 0.9628 | C / G |
| **Matched Label Accuracy** | 0.9784 | C / M (Diagnostic only, not substituted for Joint F1) |

---

## 2. Operational Statistics & Error Summary

- **Total Scheduled Answers**: 40
- **Successfully Evaluated Answers**: 40
- **Failed Answers**: 0
- **Missed Gold Claims**: 6
- **Extra Unmatched Predictions**: 4
- **Label Mismatches (Wrong Labels)**: 8

### Observed Label Mismatches:
| Claim ID | Gold Label | Predicted Label | Quote Snippet |
| :--- | :--- | :--- | :--- |
| `A04:2363:2540` | **contradicted** | **unknown** | In the 2025 India catalogue, before the current-year revision, StoneWave Ma... |
| `A06:495:697` | **unknown** | **supported** | No version of ClearTrail Plus has ever been offered in blue, including prev... |
| `A07:306:511` | **unknown** | **supported** | A retail purchase of ClearTrail Plus in Germany qualifies for the same 12-m... |
| `A12:335:515` | **unknown** | **supported** | The official India catalogue price of StoneWave Go will drop by INR 300 dur... |
| `A19:371:555` | **contradicted** | **supported** | For an authorized India purchase with the original invoice and a manufactur... |
| `A25:306:459` | **supported** | **unknown** | For the current India catalogue, CloudNote Air is listed at INR 1,500 less ... |
| `A31:324:477` | **supported** | **contradicted** | For the current India catalogue, CloudNote Studio is listed at INR 1,500 mo... |
| `A35:335:520` | **contradicted** | **supported** | With all port covers closed, ClearTrail Plus is permitted to remain immerse... |

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


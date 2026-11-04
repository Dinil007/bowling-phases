# Evaluation report

**DEVELOPMENT RESULTS - the author saw these clips while building; not an independent test.**

- Clips compared: 8  
- Excluded: 0 
- False-confident events (frame given, reference not observable): **7**
- Coverage (events output with a frame): 0.646
- Frame-level phase agreement: 0.249

## Event timing error (only where both prediction and reference give a frame)

| Event | n | median fr | max fr | median ms | max ms | <=1 | <=2 | <=3 |
|---|---|---|---|---|---|---|---|---|
| RUS | 4 | 19.5 | 429 | 650.0 | 14300.0 | 0.0 | 0.0 | 0.0 |
| GTO | 2 | 5.5 | 9 | 183.3 | 300.0 | 0.0 | 50.0 | 50.0 |
| BFC | 2 | 2.0 | 3 | 66.7 | 100.0 | 50.0 | 50.0 | 100.0 |
| FFC | 5 | 1.0 | 6 | 33.3 | 200.0 | 60.0 | 80.0 | 80.0 |
| BR | 5 | 1.0 | 4 | 43.5 | 133.3 | 60.0 | 80.0 | 80.0 |
| FTE | 6 | 19.0 | 363 | 633.3 | 12100.0 | 0.0 | 0.0 | 0.0 |

## Observability confusion (per event)

| Event | both observable | pred obs / ref not (false-confident) | pred not / ref obs | both not |
|---|---|---|---|---|
| RUS | 4 | 4 | 0 | 0 |
| GTO | 2 | 1 | 4 | 1 |
| BFC | 2 | 1 | 5 | 0 |
| FFC | 5 | 1 | 1 | 1 |
| BR | 5 | 0 | 2 | 1 |
| FTE | 6 | 0 | 0 | 2 |

False-confident cases: dev_01 RUS @ 78; dev_03 RUS @ 258; dev_03 GTO @ 269; dev_03 BFC @ 272; dev_03 FFC @ 274; dev_04 RUS @ 54; dev_05 RUS @ 100

## Per-clip results (pred / ref, error in frames)

| Clip | RUS | GTO | BFC | FFC | BR | FTE | phase agr. | coverage |
|---|---|---|---|---|---|---|---|---|
| dev_01 | 78/None | None/None | None/84 | 83/89 (-6) | 90/91 (-1) | 96/459 (-363) | 0.011 | 0.667 |
| dev_02 | 432/3 (+429) | None/20 | None/25 | None/28 | None/31 | None/None | 0.0 | 0.167 |
| dev_03 | 258/None | 269/None | 272/None | 274/None | None/None | None/None | None | 0.667 |
| dev_04 | 54/None | None/57 [ref in win] | None/60 [ref in win] | 65/63 (+2) | 67/65 (+2) | 79/100 (-21) | 0.256 | 0.667 |
| dev_05 | 100/None | None/100 [ref in win] | None/102 [ref in win] | None/None | 106/107 (-1) | 125/130 (-5) | 0.567 | 0.5 |
| dev_06 | 36/6 (+30) | 136/127 (+9) | 137/134 (+3) | 139/138 (+1) | None/141 [ref in win] | 148/170 (-22) | 0.555 | 0.833 |
| dev_07 | 84/75 (+9) | 161/159 (+2) | 162/163 (-1) | 167/167 (+0) | 170/170 (+0) | 176/192 (-16) | 0.761 | 1.0 |
| dev_08 | 36/31 (+5) | None/106 [ref in win] | None/110 [ref in win] | 114/115 (-1) | 120/116 (+4) | 126/143 (-17) | 0.045 | 0.667 |

## Phase confusion (rows = reference, cols = prediction, frames)

| ref \ pred | Run-up | Gather | Delivery stride / release preparation | Ball release | Follow-through | none |
|---|---|---|---|---|---|---|
| Run-up | 166 | 0 | 0 | 0 | 0 | 131 |
| Gather | 9 | 1 | 1 | 0 | 0 | 14 |
| Delivery stride / release preparation | 2 | 1 | 7 | 2 | 0 | 31 |
| Ball release | 0 | 0 | 0 | 1 | 2 | 4 |
| Follow-through | 0 | 0 | 0 | 2 | 42 | 454 |

## Robustness and processing time

- {'total': 8, 'ok': 8, 'failed': 0}
- Hardware: {'machine': 'AMD64', 'processor': 'AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD', 'cpus': 16, 'system': 'Windows-11-10.0.26300-SP0', 'python': '3.12.4'}

| Clip | wall s | s per video s |
|---|---|---|
| dev_01 | 188.06 | 6.661 |
| dev_02 | 204.04 | 4.382 |
| dev_03 | 258.86 | 4.184 |
| dev_04 | 105.81 | 19.474 |
| dev_05 | 122.28 | 12.281 |
| dev_06 | 127.52 | 10.627 |
| dev_07 | 214.42 | 12.814 |
| dev_08 | 193.95 | 11.972 |

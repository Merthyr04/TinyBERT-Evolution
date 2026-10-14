# Deployment-side benchmark (round 75)

Threshold theta = 0.5; batch size 16; 12 rounds; 416 paragraphs per pass.


## GPU round-robin (baseline = S0)

Wall clock of the timed region: 21.7 s.  SM clock 1935.0-1995.0 MHz (mean 1982.1); temperature mean 60.3 C, max 66.0 C; board power mean 80.19 W, max 111.56 W; 45 samples.
Energy over the timed region: 1706.0 J (341.75 J per 1000 paragraphs per round of the whole table).

| unit | passages/s | speed index | batch-1 p50 (ms) | batch-1 p99 (ms) |
|---|---|---|---|---|
| F2@jit | 7406.75 | 334.63 | 1.876 | 3.179 |
| S0@jit | 3845.14 | 173.72 | 3.153 | 4.682 |
| F2 | 3763.96 | 170.05 | 3.932 | 6.261 |
| S7@exit | 3346.91 | 151.21 | 4.23 | 6.632 |
| S0 | 2213.42 | 100.0 | 6.838 | 9.703 |
| S7@noexit | 2003.08 | 90.5 | 7.116 | 10.745 |

## CPU round-robin (baseline = S0, 8 threads, 4 rounds)

| unit | passages/s | speed index | batch-1 p50 (ms) | batch-1 p99 (ms) |
|---|---|---|---|---|
| F2 | 454.01 | 200.21 | 4.288 | 5.546 |
| S7@exit | 443.92 | 195.76 | 4.425 | 5.99 |
| S7@int8 | 287.48 | 126.77 | 12.147 | 14.913 |
| F2@int8 | 286.51 | 126.35 | 12.323 | 15.253 |
| S0@int8 | 286.12 | 126.17 | 11.983 | 15.008 |
| S7@noexit | 229.17 | 101.06 | 8.167 | 10.242 |
| S0 | 226.77 | 100.0 | 8.184 | 9.903 |

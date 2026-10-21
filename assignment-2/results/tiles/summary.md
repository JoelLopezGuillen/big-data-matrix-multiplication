# Summary

| Method | Repr. | Phase | n | Tile | Density % | Status | Reps | Median (s) | IQR (s) | Q1 (s) | Q3 (s) | Peak WS (MiB) | nnz C | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | dense | multiply | 1024 | 16 |  | ok | 5 | 0.238263 | 0.00237 | 0.236662 | 0.239037 | 27.8 |  |  |
| blocked | dense | multiply | 2048 | 16 |  | ok | 5 | 3.26757 | 0.134 | 3.25614 | 3.38982 | 99.8 |  |  |
| blocked | dense | multiply | 1024 | 32 |  | ok | 5 | 0.202118 | 0.000639 | 0.20176 | 0.202399 | 27.8 |  |  |
| blocked | dense | multiply | 2048 | 32 |  | ok | 5 | 2.91676 | 0.0086 | 2.91017 | 2.91878 | 99.8 |  |  |
| blocked | dense | multiply | 1024 | 64 |  | ok | 5 | 0.213304 | 0.000497 | 0.213005 | 0.213503 | 27.8 |  |  |
| blocked | dense | multiply | 2048 | 64 |  | ok | 5 | 2.78445 | 0.0376 | 2.76729 | 2.80493 | 99.8 |  |  |
| blocked | dense | multiply | 1024 | 128 |  | ok | 5 | 0.192531 | 0.00278 | 0.190601 | 0.193379 | 27.8 |  |  |
| blocked | dense | multiply | 2048 | 128 |  | ok | 5 | 2.57537 | 0.0447 | 2.56707 | 2.6118 | 99.8 |  |  |
| blocked | dense | multiply | 1024 | 256 |  | ok | 5 | 0.231321 | 0.000906 | 0.230995 | 0.231901 | 27.8 |  |  |
| blocked | dense | multiply | 2048 | 256 |  | ok | 5 | 2.98279 | 0.0947 | 2.92293 | 3.01766 | 99.8 |  |  |

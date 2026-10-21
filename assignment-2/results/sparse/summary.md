# Summary

| Method | Repr. | Phase | n | Tile | Density % | Status | Reps | Median (s) | IQR (s) | Q1 (s) | Q3 (s) | Peak WS (MiB) | nnz C | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | dense | multiply | 1000 | 128 | 0.1 | ok | 5 | 0.187096 | 0.0246 | 0.180252 | 0.20485 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 0.1 | ok | 5 | 1.44648 | 0.0756 | 1.39909 | 1.47467 | 95.4 |  |  |
| blocked | dense | multiply | 4000 | 128 | 0.1 | ok | 5 | 11.2413 | 0.0227 | 11.22 | 11.2427 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 0.1 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| blocked | dense | multiply | 1000 | 128 | 1 | ok | 5 | 0.17642 | 0.000975 | 0.175932 | 0.176907 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 1 | ok | 5 | 1.40583 | 0.00993 | 1.40234 | 1.41227 | 95.3 |  |  |
| blocked | dense | multiply | 4000 | 128 | 1 | ok | 5 | 11.2239 | 0.0131 | 11.2201 | 11.2333 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 1 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| blocked | dense | multiply | 1000 | 128 | 5 | ok | 5 | 0.179258 | 0.00284 | 0.177733 | 0.180576 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 5 | ok | 5 | 1.41355 | 0.00311 | 1.41311 | 1.41622 | 95.3 |  |  |
| blocked | dense | multiply | 4000 | 128 | 5 | ok | 5 | 11.2469 | 0.147 | 11.1927 | 11.3396 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 5 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| blocked | dense | multiply | 1000 | 128 | 10 | ok | 5 | 0.177229 | 0.00105 | 0.176599 | 0.177647 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 10 | ok | 5 | 1.42031 | 0.0168 | 1.41147 | 1.42832 | 95.3 |  |  |
| blocked | dense | multiply | 4000 | 128 | 10 | ok | 5 | 11.1911 | 0.067 | 11.1243 | 11.1913 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 10 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| blocked | dense | multiply | 1000 | 128 | 25 | ok | 5 | 0.174511 | 0.000429 | 0.174253 | 0.174682 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 25 | ok | 5 | 1.42688 | 0.0342 | 1.39412 | 1.42827 | 95.3 |  |  |
| blocked | dense | multiply | 4000 | 128 | 25 | ok | 5 | 11.2497 | 0.0266 | 11.2246 | 11.2512 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 25 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| blocked | dense | multiply | 1000 | 128 | 50 | ok | 5 | 0.176291 | 0.00168 | 0.175509 | 0.177188 | 26.7 |  |  |
| blocked | dense | multiply | 2000 | 128 | 50 | ok | 5 | 1.39993 | 0.00792 | 1.39646 | 1.40438 | 95.4 |  |  |
| blocked | dense | multiply | 4000 | 128 | 50 | ok | 5 | 11.1736 | 0.0525 | 11.1513 | 11.2038 | 370.0 |  |  |
| blocked | dense | multiply | 10000 | 128 | 50 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| csr_gustavson | csr | multiply | 1000 |  | 0.1 | ok | 5 | 9.2e-06 | 6.7e-06 | 7.2e-06 | 1.39e-05 | 3.9 | 958 |  |
| csr_gustavson | csr | multiply | 2000 |  | 0.1 | ok | 5 | 0.0002014 | 1.64e-05 | 0.0002008 | 0.0002172 | 4.0 | 7868 |  |
| csr_gustavson | csr | multiply | 4000 |  | 0.1 | ok | 5 | 0.0017573 | 2.23e-05 | 0.0017361 | 0.0017584 | 5.0 | 64204 |  |
| csr_gustavson | csr | multiply | 10000 |  | 0.1 | ok | 5 | 0.0409822 | 0.000398 | 0.0407614 | 0.0411595 | 17.8 | 994450 |  |
| csr_gustavson | csr | multiply | 1000 |  | 1 | ok | 5 | 0.0014396 | 1.62e-05 | 0.001425 | 0.0014412 | 5.2 | 95440 |  |
| csr_gustavson | csr | multiply | 2000 |  | 1 | ok | 5 | 0.0099083 | 3.24e-05 | 0.0099019 | 0.0099343 | 13.1 | 724188 |  |
| csr_gustavson | csr | multiply | 4000 |  | 1 | ok | 5 | 0.0744911 | 0.00116 | 0.0742603 | 0.0754245 | 68.0 | 5275068 |  |
| csr_gustavson | csr | multiply | 10000 |  | 1 | ok | 5 | 1.07448 | 0.0232 | 1.0555 | 1.07872 | 750.4 | 63211362 |  |
| csr_gustavson | csr | multiply | 1000 |  | 5 | ok | 5 | 0.0157315 | 9.39e-05 | 0.0157271 | 0.015821 | 15.5 | 918565 |  |
| csr_gustavson | csr | multiply | 2000 |  | 5 | ok | 5 | 0.0808803 | 0.000995 | 0.0801481 | 0.0811431 | 53.9 | 3973101 |  |
| csr_gustavson | csr | multiply | 4000 |  | 5 | ok | 5 | 0.415529 | 0.00742 | 0.408591 | 0.416015 | 205.3 | 15999280 |  |
| csr_gustavson | csr | multiply | 10000 |  | 5 | ok | 5 | 5.44854 | 0.02 | 5.43246 | 5.45248 | 1263.0 | 100000000 |  |
| csr_gustavson | csr | multiply | 1000 |  | 10 | ok | 5 | 0.023917 | 2.45e-05 | 0.0238962 | 0.0239207 | 17.6 | 999958 |  |
| csr_gustavson | csr | multiply | 2000 |  | 10 | ok | 5 | 0.136182 | 0.00154 | 0.135375 | 0.136914 | 58.8 | 4000000 |  |
| csr_gustavson | csr | multiply | 4000 |  | 10 | ok | 5 | 0.888766 | 0.0229 | 0.883918 | 0.906771 | 223.7 | 16000000 |  |
| csr_gustavson | csr | multiply | 10000 |  | 10 | ok | 5 | 16.1871 | 0.643 | 15.9058 | 16.5489 | 1377.4 | 100000000 |  |
| csr_gustavson | csr | multiply | 1000 |  | 25 | ok | 5 | 0.0663066 | 0.00151 | 0.0655799 | 0.067086 | 21.0 | 1000000 |  |
| csr_gustavson | csr | multiply | 2000 |  | 25 | ok | 5 | 0.495941 | 0.0195 | 0.483373 | 0.502859 | 72.5 | 4000000 |  |
| csr_gustavson | csr | multiply | 4000 |  | 25 | ok | 5 | 4.84412 | 0.0582 | 4.82491 | 4.88313 | 278.6 | 16000000 |  |
| csr_gustavson | csr | multiply | 10000 |  | 25 | timeout |  |  |  |  |  |  |  | a single kernel execution exceeded 60 s (after 0 completed executions) |
| csr_gustavson | csr | multiply | 1000 |  | 50 | ok | 5 | 0.216048 | 0.00625 | 0.214026 | 0.220273 | 26.7 | 1000000 |  |
| csr_gustavson | csr | multiply | 2000 |  | 50 | ok | 5 | 1.82314 | 0.0982 | 1.72532 | 1.82356 | 95.4 | 4000000 |  |
| csr_gustavson | csr | multiply | 4000 |  | 50 | ok | 5 | 17.9534 | 0.133 | 17.9008 | 18.034 | 370.1 | 16000000 |  |
| csr_gustavson | csr | multiply | 10000 |  | 50 | timeout |  |  |  |  |  |  |  | a single kernel execution exceeded 60 s (after 0 completed executions) |
| dense_to_csr | csr | convert | 1000 |  | 0.1 | ok | 5 | 0.0022073 | 5.26e-05 | 0.002187 | 0.0022396 | 19.1 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 0.1 | ok | 5 | 0.0116892 | 0.000417 | 0.0116125 | 0.0120298 | 64.9 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 0.1 | ok | 5 | 0.0398274 | 0.00226 | 0.0397278 | 0.0419878 | 248.4 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 0.1 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| dense_to_csr | csr | convert | 1000 |  | 1 | ok | 5 | 0.0021069 | 0.00019 | 0.002063 | 0.0022527 | 19.3 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 1 | ok | 5 | 0.0099326 | 8e-06 | 0.0099325 | 0.0099405 | 66.0 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 1 | ok | 5 | 0.0482759 | 0.000278 | 0.0480178 | 0.0482961 | 252.2 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 1 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| dense_to_csr | csr | convert | 1000 |  | 5 | ok | 5 | 0.0029106 | 5.93e-05 | 0.0028811 | 0.0029404 | 20.4 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 5 | ok | 5 | 0.0146915 | 0.000499 | 0.0142434 | 0.0147428 | 70.2 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 5 | ok | 5 | 0.0649884 | 0.00131 | 0.0647698 | 0.0660827 | 266.3 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 5 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| dense_to_csr | csr | convert | 1000 |  | 10 | ok | 5 | 0.0040505 | 0.000294 | 0.0039627 | 0.004257 | 21.8 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 10 | ok | 5 | 0.019681 | 0.000515 | 0.0196605 | 0.0201757 | 74.0 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 10 | ok | 5 | 0.0778751 | 0.00385 | 0.0765744 | 0.0804214 | 284.6 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 10 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| dense_to_csr | csr | convert | 1000 |  | 25 | ok | 5 | 0.0064372 | 5.5e-05 | 0.0064343 | 0.0064893 | 24.8 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 25 | ok | 5 | 0.0282245 | 0.000375 | 0.0279604 | 0.0283351 | 87.8 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 25 | ok | 5 | 0.118841 | 0.00286 | 0.116144 | 0.119005 | 339.5 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 25 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |
| dense_to_csr | csr | convert | 1000 |  | 50 | ok | 5 | 0.0100642 | 8.99e-05 | 0.0100165 | 0.0101064 | 30.5 |  |  |
| dense_to_csr | csr | convert | 2000 |  | 50 | ok | 5 | 0.0414584 | 0.000126 | 0.0413597 | 0.041486 | 110.6 |  |  |
| dense_to_csr | csr | convert | 4000 |  | 50 | ok | 5 | 0.161013 | 0.00141 | 0.160824 | 0.162238 | 431.1 |  |  |
| dense_to_csr | csr | convert | 10000 |  | 50 | not_run |  |  |  |  |  |  |  | not run by design: n=10000 is a CSR-only size (dense comparison sizes: [1000, 2000, 4000]) |


# Sparse vs dense

End-to-end = median dense->CSR conversion (A and B) + median SpGEMM (a derived value, not a measured run).


| n | Density % | nnz A | nnz B | nnz C | Output density % | SpGEMM (s) | Convert A+B (s) | End-to-end (s) | Dense blocked (s) | Dense / SpGEMM | Dense / end-to-end | SpGEMM status | Dense status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.1 | 1000 | 1000 | 958 | 0.0958 | 9.2e-06 | 0.0022073 | 0.0022165 | 0.187096 | 2.03e+04 | 84.4 | ok | ok |
| 1000 | 1 | 10000 | 10000 | 95440 | 9.544 | 0.0014396 | 0.0021069 | 0.0035465 | 0.17642 | 123 | 49.7 | ok | ok |
| 1000 | 5 | 50000 | 50000 | 918565 | 91.86 | 0.0157315 | 0.0029106 | 0.0186421 | 0.179258 | 11.4 | 9.62 | ok | ok |
| 1000 | 10 | 100000 | 100000 | 999958 | 100 | 0.023917 | 0.0040505 | 0.0279675 | 0.177229 | 7.41 | 6.34 | ok | ok |
| 1000 | 25 | 250000 | 250000 | 1000000 | 100 | 0.0663066 | 0.0064372 | 0.0727438 | 0.174511 | 2.63 | 2.4 | ok | ok |
| 1000 | 50 | 500000 | 500000 | 1000000 | 100 | 0.216048 | 0.0100642 | 0.226112 | 0.176291 | 0.816 | 0.78 | ok | ok |
| 2000 | 0.1 | 4000 | 4000 | 7868 | 0.1967 | 0.0002014 | 0.0116892 | 0.0118906 | 1.44648 | 7.18e+03 | 122 | ok | ok |
| 2000 | 1 | 40000 | 40000 | 724188 | 18.1 | 0.0099083 | 0.0099326 | 0.0198409 | 1.40583 | 142 | 70.9 | ok | ok |
| 2000 | 5 | 200000 | 200000 | 3973101 | 99.33 | 0.0808803 | 0.0146915 | 0.0955718 | 1.41355 | 17.5 | 14.8 | ok | ok |
| 2000 | 10 | 400000 | 400000 | 4000000 | 100 | 0.136182 | 0.019681 | 0.155863 | 1.42031 | 10.4 | 9.11 | ok | ok |
| 2000 | 25 | 1000000 | 1000000 | 4000000 | 100 | 0.495941 | 0.0282245 | 0.524165 | 1.42688 | 2.88 | 2.72 | ok | ok |
| 2000 | 50 | 2000000 | 2000000 | 4000000 | 100 | 1.82314 | 0.0414584 | 1.8646 | 1.39993 | 0.768 | 0.751 | ok | ok |
| 4000 | 0.1 | 16000 | 16000 | 64204 | 0.4013 | 0.0017573 | 0.0398274 | 0.0415847 | 11.2413 | 6.4e+03 | 270 | ok | ok |
| 4000 | 1 | 160000 | 160000 | 5275068 | 32.97 | 0.0744911 | 0.0482759 | 0.122767 | 11.2239 | 151 | 91.4 | ok | ok |
| 4000 | 5 | 800000 | 800000 | 15999280 | 100 | 0.415529 | 0.0649884 | 0.480517 | 11.2469 | 27.1 | 23.4 | ok | ok |
| 4000 | 10 | 1600000 | 1600000 | 16000000 | 100 | 0.888766 | 0.0778751 | 0.966641 | 11.1911 | 12.6 | 11.6 | ok | ok |
| 4000 | 25 | 4000000 | 4000000 | 16000000 | 100 | 4.84412 | 0.118841 | 4.96296 | 11.2497 | 2.32 | 2.27 | ok | ok |
| 4000 | 50 | 8000000 | 8000000 | 16000000 | 100 | 17.9534 | 0.161013 | 18.1144 | 11.1736 | 0.622 | 0.617 | ok | ok |
| 10000 | 0.1 | 100000 | 100000 | 994450 | 0.9944 | 0.0409822 |  |  |  |  |  | ok | not_run |
| 10000 | 1 | 1000000 | 1000000 | 63211362 | 63.21 | 1.07448 |  |  |  |  |  | ok | not_run |
| 10000 | 5 | 5000000 | 5000000 | 100000000 | 100 | 5.44854 |  |  |  |  |  | ok | not_run |
| 10000 | 10 | 10000000 | 10000000 | 100000000 | 100 | 16.1871 |  |  |  |  |  | ok | not_run |
| 10000 | 25 | 25000000 | 25000000 |  |  |  |  |  |  |  |  | timeout | not_run |
| 10000 | 50 | 50000000 | 50000000 |  |  |  |  |  |  |  |  | timeout | not_run |

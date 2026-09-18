truth model on 3331 prefill steps: w=[6.761, -1.602, 6.28, -1.504, 0.57, 0.083, 1.672] MAE 3.8 ms, resid q95 10.1 ms; controllers fit on 2498 one-prefill / 603 multi-prefill steps; margins {'m0-one': 13.49, 'm1-one': 7.3, 'm2-one': 13.21, 'm0-multi': 55.99, 'm1-multi': 5.2, 'm2-multi': 4.05}

## N1-L16384-B8-D50  (1 prefill requests of 16384 tokens, 8 decoders, deadline 50 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 34243 | 21 / 6 / 73 |
| fixed-128 | 21.2% | 2084 | 6205 | 16 / 5 / 79 |
| fixed-256 | 74.1% | 1039 | 4088 | 50 / 50 / 0 |
| fixed-512 | 100.0% | 0 | 2660 | 100 / 0 / 0 |
| fixed-1024 | 100.0% | 0 | 2306 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 2183 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 2041 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| m0-one | 72.8% | 326 | 20725 | 30 / 4 / 66 |
| m1-one | 24.9% | 2181 | 5590 | 27 / 1 / 72 |
| m2-one | 15.1% | 10 | 679827 | 13 / 11 / 75 |
| m0-multi | 0.0% | 0 | 678192 | 0 / 0 / 0 |
| m1-multi | 58.0% | 772 | 11584 | 33 / 5 / 61 |
| m2-multi | 28.9% | 2096 | 5384 | 28 / 8 / 64 |

## N1-L16384-B8-D100  (1 prefill requests of 16384 tokens, 8 decoders, deadline 100 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 34243 | 21 / 6 / 73 |
| fixed-128 | 8.8% | 2414 | 6205 | 16 / 5 / 79 |
| fixed-256 | 10.6% | 3589 | 4088 | 50 / 50 / 0 |
| fixed-512 | 18.8% | 5006 | 2660 | 100 / 0 / 0 |
| fixed-1024 | 92.5% | 533 | 2306 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 2183 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 2041 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| m0-one | 7.7% | 4592 | 3374 | 60 / 40 / 0 |
| m1-one | 4.9% | 5178 | 3062 | 76 / 24 / 0 |
| m2-one | 7.7% | 4592 | 3374 | 60 / 40 / 0 |
| m0-multi | 54.5% | 11 | 687586 | 25 / 5 / 70 |
| m1-multi | 4.5% | 5265 | 3024 | 75 / 25 / 0 |
| m2-multi | 5.1% | 5288 | 2970 | 85 / 15 / 0 |

## N1-L16384-B8-D200  (1 prefill requests of 16384 tokens, 8 decoders, deadline 200 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 2.5% | 467 | 34243 | 21 / 6 / 73 |
| fixed-128 | 0.0% | 2644 | 6205 | 16 / 5 / 79 |
| fixed-256 | 0.0% | 4011 | 4088 | 50 / 50 / 0 |
| fixed-512 | 0.0% | 6160 | 2660 | 100 / 0 / 0 |
| fixed-1024 | 0.0% | 7105 | 2306 | 100 / 0 / 0 |
| fixed-2048 | 87.5% | 938 | 2183 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 2041 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 2091 | 0 / 0 / 100 |
| m0-one | 0.0% | 7012 | 2337 | 100 / 0 / 0 |
| m1-one | 0.0% | 7152 | 2291 | 100 / 0 / 0 |
| m2-one | 0.0% | 7012 | 2337 | 100 / 0 / 0 |
| m0-multi | 0.0% | 7105 | 2306 | 100 / 0 / 0 |
| m1-multi | 7.1% | 6290 | 2279 | 100 / 0 / 0 |
| m2-multi | 0.0% | 7152 | 2291 | 100 / 0 / 0 |

## N2-L16384-B8-D50  (2 prefill requests of 16384 tokens, 8 decoders, deadline 50 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 69246 | 0 / 6 / 94 |
| fixed-128 | 23.4% | 2022 | 12433 | 0 / 6 / 94 |
| fixed-256 | 74.7% | 1002 | 8283 | 1 / 5 / 94 |
| fixed-512 | 100.0% | 0 | 5397 | 53 / 47 / 0 |
| fixed-1024 | 100.0% | 0 | 4628 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 4377 | 6 / 0 / 94 |
| fixed-4096 | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 4085 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| m0-one | 79.1% | 10 | 701919 | 0 / 6 / 94 |
| m1-one | 51.3% | 926 | 20987 | 0 / 3 / 97 |
| m2-one | 14.0% | 15 | 680643 | 0 / 17 / 83 |
| m0-multi | 0.0% | 0 | 678192 | 0 / 0 / 0 |
| m1-multi | 56.1% | 833 | 21454 | 0 / 3 / 97 |
| m2-multi | 30.9% | 2010 | 10911 | 0 / 3 / 96 |

## N2-L16384-B8-D100  (2 prefill requests of 16384 tokens, 8 decoders, deadline 100 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 69246 | 0 / 6 / 94 |
| fixed-128 | 8.0% | 2427 | 12433 | 0 / 6 / 94 |
| fixed-256 | 11.1% | 3523 | 8283 | 1 / 5 / 94 |
| fixed-512 | 21.2% | 4783 | 5397 | 53 / 47 / 0 |
| fixed-1024 | 91.2% | 619 | 4628 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 4377 | 6 / 0 / 94 |
| fixed-4096 | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 4085 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| m0-one | 9.3% | 3647 | 8269 | 7 / 6 / 87 |
| m1-one | 9.1% | 4573 | 6528 | 31 / 8 / 60 |
| m2-one | 7.5% | 4520 | 6851 | 24 / 9 / 67 |
| m0-multi | 53.8% | 11 | 687354 | 0 / 12 / 88 |
| m1-multi | 6.6% | 5053 | 6147 | 33 / 22 / 46 |
| m2-multi | 5.8% | 5148 | 6095 | 34 / 25 / 41 |

## N2-L16384-B8-D200  (2 prefill requests of 16384 tokens, 8 decoders, deadline 200 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 2.1% | 463 | 69246 | 0 / 6 / 94 |
| fixed-128 | 0.0% | 2637 | 12433 | 0 / 6 / 94 |
| fixed-256 | 0.0% | 3959 | 8283 | 1 / 5 / 94 |
| fixed-512 | 0.0% | 6072 | 5397 | 53 / 47 / 0 |
| fixed-1024 | 0.6% | 7036 | 4628 | 100 / 0 / 0 |
| fixed-2048 | 83.8% | 1217 | 4377 | 6 / 0 / 94 |
| fixed-4096 | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 4085 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 4180 | 0 / 0 / 100 |
| m0-one | 0.0% | 6488 | 5051 | 55 / 43 / 2 |
| m1-one | 3.9% | 6272 | 4768 | 67 / 25 / 8 |
| m2-one | 0.0% | 6928 | 4730 | 89 / 8 / 3 |
| m0-multi | 0.0% | 6936 | 4724 | 97 / 3 / 0 |
| m1-multi | 1.4% | 6969 | 4584 | 93 / 0 / 7 |
| m2-multi | 0.7% | 7076 | 4602 | 97 / 0 / 3 |

## N4-L16384-B8-D50  (4 prefill requests of 16384 tokens, 8 decoders, deadline 50 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 141001 | 0 / 6 / 94 |
| fixed-128 | 32.3% | 1708 | 25971 | 0 / 6 / 94 |
| fixed-256 | 78.5% | 840 | 16761 | 0 / 6 / 94 |
| fixed-512 | 100.0% | 0 | 11012 | 1 / 5 / 94 |
| fixed-1024 | 100.0% | 0 | 9360 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 8795 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 8173 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| m0-one | 79.1% | 10 | 701598 | 0 / 13 / 87 |
| m1-one | 74.7% | 33 | 737005 | 0 / 2 / 98 |
| m2-one | 13.4% | 21 | 681459 | 0 / 25 / 75 |
| m0-multi | 0.0% | 0 | 678192 | 0 / 0 / 0 |
| m1-multi | 57.2% | 894 | 35015 | 0 / 3 / 97 |
| m2-multi | 60.9% | 697 | 46707 | 0 / 3 / 97 |

## N4-L16384-B8-D100  (4 prefill requests of 16384 tokens, 8 decoders, deadline 100 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 141001 | 0 / 6 / 94 |
| fixed-128 | 8.7% | 2305 | 25971 | 0 / 6 / 94 |
| fixed-256 | 9.9% | 3523 | 16761 | 0 / 6 / 94 |
| fixed-512 | 25.5% | 4435 | 11012 | 1 / 5 / 94 |
| fixed-1024 | 91.9% | 569 | 9360 | 100 / 0 / 0 |
| fixed-2048 | 100.0% | 0 | 8795 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 8173 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| m0-one | 33.2% | 1513 | 35092 | 0 / 2 / 98 |
| m1-one | 8.6% | 3898 | 15333 | 3 / 0 / 97 |
| m2-one | 7.4% | 4310 | 14333 | 0 / 3 / 96 |
| m0-multi | 53.4% | 11 | 687238 | 0 / 25 / 75 |
| m1-multi | 7.2% | 4929 | 12408 | 4 / 0 / 96 |
| m2-multi | 5.8% | 4919 | 12759 | 2 / 1 / 97 |

## N4-L16384-B8-D200  (4 prefill requests of 16384 tokens, 8 decoders, deadline 200 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 3.2% | 450 | 141001 | 0 / 6 / 94 |
| fixed-128 | 0.0% | 2524 | 25971 | 0 / 6 / 94 |
| fixed-256 | 0.0% | 3911 | 16761 | 0 / 6 / 94 |
| fixed-512 | 0.0% | 5951 | 11012 | 1 / 5 / 94 |
| fixed-1024 | 1.9% | 6871 | 9360 | 100 / 0 / 0 |
| fixed-2048 | 85.0% | 1118 | 8795 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 8173 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 8381 | 0 / 0 / 100 |
| m0-one | 0.0% | 4997 | 13120 | 8 / 0 / 92 |
| m1-one | 0.2% | 6407 | 10165 | 30 / 0 / 70 |
| m2-one | 0.0% | 6711 | 9766 | 61 / 0 / 39 |
| m0-multi | 0.0% | 6171 | 10621 | 26 / 0 / 74 |
| m1-multi | 0.7% | 6952 | 9338 | 93 / 0 / 7 |
| m2-multi | 0.6% | 6890 | 9422 | 88 / 0 / 12 |

## N8-L16384-B8-D50  (8 prefill requests of 16384 tokens, 8 decoders, deadline 50 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 292544 | 0 / 6 / 94 |
| fixed-128 | 48.6% | 1207 | 55835 | 0 / 6 / 94 |
| fixed-256 | 81.7% | 679 | 35319 | 0 / 6 / 94 |
| fixed-512 | 100.0% | 0 | 22826 | 0 / 6 / 94 |
| fixed-1024 | 100.0% | 0 | 19056 | 1 / 5 / 94 |
| fixed-2048 | 100.0% | 0 | 17755 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 16388 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| m0-one | 79.1% | 10 | 701438 | 0 / 26 / 74 |
| m1-one | 73.7% | 38 | 741307 | 0 / 4 / 96 |
| m2-one | 13.3% | 26 | 682092 | 0 / 41 / 59 |
| m0-multi | 0.0% | 0 | 678192 | 0 / 0 / 0 |
| m1-multi | 76.5% | 650 | 43489 | 0 / 2 / 98 |
| m2-multi | 63.5% | 57 | 723817 | 0 / 4 / 96 |

## N8-L16384-B8-D100  (8 prefill requests of 16384 tokens, 8 decoders, deadline 100 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 100.0% | 0 | 292544 | 0 / 6 / 94 |
| fixed-128 | 9.0% | 2137 | 55835 | 0 / 6 / 94 |
| fixed-256 | 10.7% | 3316 | 35319 | 0 / 6 / 94 |
| fixed-512 | 33.0% | 3845 | 22826 | 0 / 6 / 94 |
| fixed-1024 | 91.9% | 559 | 19056 | 1 / 5 / 94 |
| fixed-2048 | 100.0% | 0 | 17755 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 16388 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| m0-one | 69.8% | 69 | 771114 | 0 / 2 / 98 |
| m1-one | 15.5% | 2411 | 48876 | 0 / 1 / 99 |
| m2-one | 8.4% | 3336 | 36698 | 0 / 3 / 97 |
| m0-multi | 53.4% | 11 | 687180 | 0 / 52 / 48 |
| m1-multi | 16.8% | 4148 | 25319 | 0 / 2 / 97 |
| m2-multi | 7.4% | 4446 | 27515 | 0 / 2 / 98 |

## N8-L16384-B8-D200  (8 prefill requests of 16384 tokens, 8 decoders, deadline 200 ms, partition equal)
| controller | violation rate | safe prefill tok/s | finish (ms) | coverage measured / interp / extrap |
| --- | ---: | ---: | ---: | --- |
| fixed-64 | 5.8% | 422 | 292544 | 0 / 6 / 94 |
| fixed-128 | 0.0% | 2348 | 55835 | 0 / 6 / 94 |
| fixed-256 | 0.0% | 3711 | 35319 | 0 / 6 / 94 |
| fixed-512 | 0.0% | 5742 | 22826 | 0 / 6 / 94 |
| fixed-1024 | 5.3% | 6513 | 19056 | 1 / 5 / 94 |
| fixed-2048 | 85.0% | 1107 | 17755 | 100 / 0 / 0 |
| fixed-4096 | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| fixed-8192 | 100.0% | 0 | 16388 | 0 / 0 / 100 |
| ppas-style | 100.0% | 0 | 16845 | 0 / 0 / 100 |
| m0-one | 0.0% | 3332 | 39343 | 0 / 1 / 99 |
| m1-one | 0.3% | 4799 | 26465 | 2 / 0 / 98 |
| m2-one | 0.0% | 6413 | 20438 | 2 / 0 / 98 |
| m0-multi | 1.2% | 137 | 701175 | 0 / 2 / 98 |
| m1-multi | 0.8% | 6737 | 19212 | 8 / 0 / 92 |
| m2-multi | 0.0% | 6694 | 19582 | 6 / 0 / 94 |

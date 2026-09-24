## L20-vllm0.30
  primary_geometry_one_to_multi (train 2478, test 589)
    M0   MAE 346.97  P95 1197.34  signed -339.89  [by prefill count 2:-54.1, 4:-201.6, 8:-688.3]  in-sample MAE 35.29
    M2   MAE  14.92  P95  25.54  signed  -7.78  [by prefill count 2:-4.9, 4:-2.6, 8:-16.9]  in-sample MAE 7.21
    M2n  MAE   6.30  P95   9.33  signed  +4.49  [by prefill count 2:-1.9, 4:+6.9, 8:+4.2]  in-sample MAE 7.95
  reverse_geometry_multi_to_one (train 589, test 84)
    M0   MAE  34.10  P95 107.30  signed +27.97  [by prefill count 1:+28.0]  in-sample MAE 31.68
    M2   MAE  25.88  P95  71.08  signed -15.18  [by prefill count 1:-15.2]  in-sample MAE 13.70
    M2n  MAE  17.21  P95  38.73  signed -12.85  [by prefill count 1:-12.9]  in-sample MAE 8.30

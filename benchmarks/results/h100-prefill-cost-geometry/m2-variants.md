## H100-Qwen3-4B
  primary_geometry_one_to_multi (train 2508, test 587)
    M0   MAE  56.08  P95 202.03  signed -55.61  [by prefill count 2:-8.4, 4:-32.4, 7:+5.2, 8:-118.8]  in-sample MAE 4.86
    M2   MAE   9.07  P95  26.52  signed  +8.82  [by prefill count 2:+1.2, 4:+6.2, 7:+5.9, 8:+16.8]  in-sample MAE 1.89
    M2n  MAE   2.10  P95   6.98  signed  +1.04  [by prefill count 2:-0.8, 4:+0.2, 7:+5.9, 8:+2.8]  in-sample MAE 1.81
  reverse_geometry_multi_to_one (train 587, test 94)
    M0   MAE   6.27  P95  19.41  signed  +3.76  [by prefill count 1:+3.8]  in-sample MAE 5.83
    M2   MAE   1.92  P95   9.54  signed  -0.47  [by prefill count 1:-0.5]  in-sample MAE 1.59
    M2n  MAE   1.61  P95   7.24  signed  -0.17  [by prefill count 1:-0.2]  in-sample MAE 1.57
## H100-Qwen3-8B
  primary_geometry_one_to_multi (train 2508, test 591)
    M0   MAE  56.76  P95 207.34  signed -56.09  [by prefill count 2:-7.9, 4:-32.7, 7:+9.2, 8:-121.6]  in-sample MAE 5.01
    M2   MAE   6.94  P95  18.96  signed  +6.74  [by prefill count 2:+1.6, 4:+4.8, 7:+10.1, 8:+12.1]  in-sample MAE 2.05
    M2n  MAE   2.53  P95   7.18  signed  +2.03  [by prefill count 2:+0.3, 4:+1.2, 7:+10.1, 8:+3.6]  in-sample MAE 2.01
  reverse_geometry_multi_to_one (train 591, test 94)
    M0   MAE   6.52  P95  19.29  signed  +3.64  [by prefill count 1:+3.6]  in-sample MAE 6.16
    M2   MAE   2.39  P95  11.27  signed  -0.74  [by prefill count 1:-0.7]  in-sample MAE 2.22
    M2n  MAE   2.49  P95   9.50  signed  -0.55  [by prefill count 1:-0.5]  in-sample MAE 2.26
## H100-Qwen2.5-1.5B-Instruct
  primary_geometry_one_to_multi (train 2510, test 588)
    M0   MAE  18.85  P95  64.30  signed -18.39  [by prefill count 2:-3.9, 4:-10.9, 7:+3.6, 8:-38.9]  in-sample MAE 2.15
    M2   MAE   5.60  P95  19.48  signed  +5.27  [by prefill count 2:-0.3, 4:+3.3, 7:+10.0, 8:+10.6]  in-sample MAE 1.74
    M2n  MAE   1.68  P95   2.60  signed  -0.66  [by prefill count 2:-1.8, 4:-1.3, 7:+8.6, 8:+0.1]  in-sample MAE 1.54
  reverse_geometry_multi_to_one (train 588, test 96)
    M0   MAE   3.22  P95  10.82  signed  -0.03  [by prefill count 1:-0.0]  in-sample MAE 2.62
    M2   MAE   2.39  P95  11.54  signed  -1.50  [by prefill count 1:-1.5]  in-sample MAE 1.58
    M2n  MAE   2.27  P95  10.72  signed  -1.37  [by prefill count 1:-1.4]  in-sample MAE 1.58
## H100-Qwen2.5-7B-Instruct
  primary_geometry_one_to_multi (train 2507, test 586)
    M0   MAE  44.37  P95 160.06  signed -43.97  [by prefill count 2:-6.3, 4:-25.4, 7:+6.5, 8:-94.4]  in-sample MAE 3.86
    M2   MAE   6.46  P95  18.41  signed  +6.30  [by prefill count 2:+1.3, 4:+4.2, 7:+7.1, 8:+11.8]  in-sample MAE 1.75
    M2n  MAE   2.04  P95   5.87  signed  +1.40  [by prefill count 2:-0.0, 4:+0.5, 7:+7.1, 8:+3.0]  in-sample MAE 1.67
  reverse_geometry_multi_to_one (train 586, test 94)
    M0   MAE   4.99  P95  15.27  signed  +3.00  [by prefill count 1:+3.0]  in-sample MAE 4.70
    M2   MAE   1.79  P95   7.76  signed  -0.03  [by prefill count 1:-0.0]  in-sample MAE 1.65
    M2n  MAE   1.81  P95   7.22  signed  +0.04  [by prefill count 1:+0.0]  in-sample MAE 1.65

## L20-Qwen3-4B
  primary_geometry_one_to_multi (train 2486, test 595)
    M0   MAE 339.76  P95 1200.12  signed -339.58  [by prefill count 2:-55.0, 4:-205.8, 8:-700.5]  in-sample MAE 30.68
    M2   MAE  10.35  P95  24.08  signed -10.06  [by prefill count 2:-4.6, 4:-8.6, 8:-15.6]  in-sample MAE 2.05
    M2n  MAE   3.07  P95  10.63  signed  +1.80  [by prefill count 2:-1.5, 4:+0.5, 8:+5.5]  in-sample MAE 2.43
  reverse_geometry_multi_to_one (train 595, test 87)
    M0   MAE  36.76  P95 125.76  signed +32.80  [by prefill count 1:+32.8]  in-sample MAE 33.17
    M2   MAE   4.45  P95  11.33  signed  +1.27  [by prefill count 1:+1.3]  in-sample MAE 2.51
    M2n  MAE   2.52  P95  13.35  signed  +1.91  [by prefill count 1:+1.9]  in-sample MAE 1.53
## A100-Qwen3-4B
  primary_geometry_one_to_multi (train 2502, test 589)
    M0   MAE 171.45  P95 589.33  signed -171.29  [by prefill count 2:-31.0, 4:-106.2, 7:+3.6, 8:-355.3]  in-sample MAE 16.29
    M2   MAE  18.55  P95  47.48  signed -18.30  [by prefill count 2:-7.5, 4:-14.9, 7:+5.3, 8:-30.6]  in-sample MAE 2.16
    M2n  MAE   1.92  P95   3.97  signed  -0.26  [by prefill count 2:-2.8, 4:-1.1, 7:+5.1, 8:+1.8]  in-sample MAE 1.67
  reverse_geometry_multi_to_one (train 589, test 93)
    M0   MAE  18.34  P95  62.87  signed +16.56  [by prefill count 1:+16.6]  in-sample MAE 16.47
    M2   MAE   3.34  P95  13.20  signed  +0.73  [by prefill count 1:+0.7]  in-sample MAE 2.44
    M2n  MAE   2.31  P95  14.12  signed  +1.48  [by prefill count 1:+1.5]  in-sample MAE 2.07
## A100-Qwen3-8B
  primary_geometry_one_to_multi (train 2485, test 594)
    M0   MAE 169.98  P95 592.23  signed -169.44  [by prefill count 2:-26.7, 4:-104.7, 8:-348.3]  in-sample MAE 15.70
    M2   MAE  17.98  P95  50.81  signed -17.37  [by prefill count 2:-3.7, 4:-13.5, 8:-31.4]  in-sample MAE 1.55
    M2n  MAE   1.66  P95   3.32  signed  +1.53  [by prefill count 2:+1.2, 4:+1.0, 8:+2.1]  in-sample MAE 1.02
  reverse_geometry_multi_to_one (train 594, test 88)
    M0   MAE  18.60  P95  64.66  signed +16.21  [by prefill count 1:+16.2]  in-sample MAE 16.67
    M2   MAE   2.64  P95   9.27  signed  +0.31  [by prefill count 1:+0.3]  in-sample MAE 1.83
    M2n  MAE   1.72  P95   9.70  signed  +0.63  [by prefill count 1:+0.6]  in-sample MAE 1.58
## L20-Qwen2.5-1.5B
  primary_geometry_one_to_multi (train 2500, test 590)
    M0   MAE 116.37  P95 392.59  signed -116.18  [by prefill count 2:-21.4, 4:-74.1, 7:-49.5, 8:-235.6]  in-sample MAE 11.53
    M2   MAE  23.89  P95  67.32  signed -23.69  [by prefill count 2:-7.4, 4:-18.8, 7:-6.9, 8:-41.0]  in-sample MAE 2.27
    M2n  MAE   1.75  P95   3.39  signed  -0.62  [by prefill count 2:-1.4, 4:-1.0, 7:+2.2, 8:+0.2]  in-sample MAE 1.59
  reverse_geometry_multi_to_one (train 590, test 91)
    M0   MAE  12.68  P95  42.06  signed +10.83  [by prefill count 1:+10.8]  in-sample MAE 11.36
    M2   MAE   2.86  P95   8.16  signed  -0.26  [by prefill count 1:-0.3]  in-sample MAE 1.98
    M2n  MAE   2.16  P95   6.36  signed  +0.05  [by prefill count 1:+0.0]  in-sample MAE 1.87
## A100-Qwen2.5-7B
  primary_geometry_one_to_multi (train 2486, test 589)
    M0   MAE 122.72  P95 421.66  signed -122.42  [by prefill count 2:-20.1, 4:-76.3, 8:-244.4]  in-sample MAE 11.37
    M2   MAE  16.98  P95  48.41  signed -16.57  [by prefill count 2:-4.2, 4:-13.0, 8:-28.6]  in-sample MAE 1.37
    M2n  MAE   1.23  P95   2.34  signed  +0.88  [by prefill count 2:+0.3, 4:+0.5, 8:+1.7]  in-sample MAE 0.96
  reverse_geometry_multi_to_one (train 589, test 88)
    M0   MAE  13.44  P95  44.81  signed +11.35  [by prefill count 1:+11.3]  in-sample MAE 12.11
    M2   MAE   2.32  P95   7.80  signed  -0.10  [by prefill count 1:-0.1]  in-sample MAE 1.63
    M2n  MAE   1.60  P95   7.22  signed  +0.16  [by prefill count 1:+0.2]  in-sample MAE 1.43
## A100-Qwen2.5-1.5B
  primary_geometry_one_to_multi (train 2505, test 592)
    M0   MAE  55.85  P95 183.15  signed -55.69  [by prefill count 2:-12.9, 4:-37.5, 7:+3.9, 8:-109.9]  in-sample MAE 6.24
    M2   MAE  17.36  P95  45.91  signed -17.18  [by prefill count 2:-7.0, 4:-14.2, 7:+4.3, 8:-28.5]  in-sample MAE 2.05
    M2n  MAE   2.13  P95   3.70  signed  -1.73  [by prefill count 2:-3.0, 4:-2.2, 7:+4.1, 8:-0.7]  in-sample MAE 1.66
  reverse_geometry_multi_to_one (train 592, test 93)
    M0   MAE   6.32  P95  20.12  signed  +5.28  [by prefill count 1:+5.3]  in-sample MAE 5.64
    M2   MAE   1.90  P95   8.67  signed  +0.27  [by prefill count 1:+0.3]  in-sample MAE 1.45
    M2n  MAE   1.48  P95   8.92  signed  +0.47  [by prefill count 1:+0.5]  in-sample MAE 1.40

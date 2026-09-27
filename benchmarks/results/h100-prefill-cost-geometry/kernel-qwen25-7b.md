## fa2 (qwen25-7b, NVIDIA H100 80GB HBM3): W 79.7us + 0.0480 ms/M | split c_S/c_X 0.36 | TS bq 64 bk 176 grid MAE 32.0 us
   k4k/16k q256/768                   meas   +316.6 us  W   +301.9  TS   +307.3
   k4k/16k q128/896                   meas   +340.8 us  W   +452.8  TS   +441.5
   k4k/16k q384/640                   meas   +223.2 us  W   +150.9  TS   +234.1
   k8k/24k q256/768                   meas   +417.1 us  W   +402.5  TS   +343.9
   k0/16k q256/768                    meas   +434.9 us  W   +402.5  TS   +468.3
   k4k/16k q512/512 (control, dW=0)   meas     -1.9 us  W     +0.0  TS     +0.0
   HK1 True  HK3 False  HK4 MAE W 49.2 vs TS 45.5 us  HK5 W-over 1/5 TS-within 5/5
## fa3 (qwen25-7b, NVIDIA H100 80GB HBM3): W 25.8us + 0.0258 ms/M | split c_S/c_X 0.69 | TS bq 64 bk 176 grid MAE 24.4 us
   k4k/16k q256/768                   meas   +154.8 us  W   +162.3  TS   +163.9
   k4k/16k q128/896                   meas   +158.0 us  W   +243.4  TS   +235.4
   k4k/16k q384/640                   meas   +148.2 us  W    +81.1  TS   +124.8
   k8k/24k q256/768                   meas   +207.1 us  W   +216.4  TS   +183.4
   k0/16k q256/768                    meas   +208.4 us  W   +216.4  TS   +249.7
   k4k/16k q512/512 (control, dW=0)   meas     +0.2 us  W     +0.0  TS     +0.0
   HK1 True  HK3 False  HK4 MAE W 35.5 vs TS 35.0 us  HK5 W-over 1/5 TS-within 4/5
## flashinfer (qwen25-7b, NVIDIA H100 80GB HBM3): W 48.7us + 0.0277 ms/M | split c_S/c_X 0.34 | TS bq 128 bk 176 grid MAE 16.9 us
   k4k/16k q256/768                   meas   +157.5 us  W   +174.1  TS   +243.5
   k4k/16k q128/896                   meas   +174.7 us  W   +261.1  TS   +251.2
   k4k/16k q384/640                   meas   +109.6 us  W    +87.0  TS   +235.8
   k8k/24k q256/768                   meas   +215.9 us  W   +232.1  TS   +351.2
   k0/16k q256/768                    meas   +225.5 us  W   +232.1  TS   +243.5
   k4k/16k q512/512 (control, dW=0)   meas     -0.4 us  W     +0.0  TS     +0.0
   HK1 True  HK3 False  HK4 MAE W 29.7 vs TS 88.4 us  HK5 W-over 1/5 TS-within 1/5

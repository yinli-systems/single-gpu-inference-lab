## fa2 (qwen3-4b, NVIDIA H100 80GB HBM3): W 65.2us + 0.0526 ms/M | split c_S/c_X 0.78 | TS bq 64 bk 176 grid MAE 27.2 us
   k4k/16k q256/768                   meas   +260.8 us  W   +330.7  TS   +331.8  closure 1.25
   k4k/16k q128/896                   meas   +335.6 us  W   +496.0  TS   +572.9  closure 1.58
   k4k/16k q384/640                   meas   +200.4 us  W   +165.3  TS   +174.3  closure 1.36
   k8k/24k q256/768                   meas   +295.7 us  W   +440.9  TS   +444.0  closure 1.05
   k0/16k q256/768                    meas   +433.5 us  W   +440.9  TS   +444.0  closure 2.04
   k4k/16k q512/512 (control, dW=0)   meas     -0.8 us  W     +0.0  TS     +0.0  closure 0.26
   HK1 True  HK3 False  HK4 MAE W 83.6 vs TS 98.6 us  HK5 W-over 3/5 TS-within 3/5
## fa3 (qwen3-4b, NVIDIA H100 80GB HBM3): W 16.0us + 0.0270 ms/M | split c_S/c_X 1.21 | TS bq 64 bk 176 grid MAE 16.1 us
   k4k/16k q256/768                   meas   +155.7 us  W   +170.1  TS   +169.9  closure 0.75
   k4k/16k q128/896                   meas   +156.5 us  W   +255.2  TS   +293.3  closure 0.74
   k4k/16k q384/640                   meas   +149.9 us  W    +85.1  TS    +89.2  closure 1.02
   k8k/24k q256/768                   meas   +204.1 us  W   +226.8  TS   +227.3  closure 0.72
   k0/16k q256/768                    meas   +205.0 us  W   +226.8  TS   +227.3  closure 0.97
   k4k/16k q512/512 (control, dW=0)   meas     +0.2 us  W     +0.0  TS     +0.0  closure -0.07
   HK1 True  HK3 True  HK4 MAE W 44.5 vs TS 51.4 us  HK5 W-over 2/5 TS-within 3/5
## flashinfer (qwen3-4b, NVIDIA H100 80GB HBM3): W 51.2us + 0.0291 ms/M | split c_S/c_X 0.62 | TS bq 128 bk 176 grid MAE 18.9 us
   k4k/16k q256/768                   meas   +115.2 us  W   +183.4  TS   +257.3  closure 0.55
   k4k/16k q128/896                   meas   +135.8 us  W   +275.1  TS   +262.7  closure 0.64
   k4k/16k q384/640                   meas    +97.3 us  W    +91.7  TS   +190.3  closure 0.66
   k8k/24k q256/768                   meas   +116.7 us  W   +244.5  TS   +361.8  closure 0.41
   k0/16k q256/768                    meas   +225.3 us  W   +244.5  TS   +257.3  closure 1.06
   k4k/16k q512/512 (control, dW=0)   meas     +0.3 us  W     +0.0  TS     +0.0  closure -0.10
   HK1 True  HK3 False  HK4 MAE W 72.0 vs TS 127.8 us  HK5 W-over 3/5 TS-within 1/5

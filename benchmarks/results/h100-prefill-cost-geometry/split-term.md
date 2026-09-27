## H100-Qwen3-4B: c_X 0.956  c_S 5.721  c_S/c_X 5.99  single slope 1.067 (spread -6.2%..+4.8%)
   primary_geometry_one_to_multi: M0 56.08  M2n 2.10  M2s 12.17
   reverse_geometry_multi_to_one: M0 6.27  M2n 1.61  M2s 4.33
   k4k/16k q256/768                   measured  +7.49  split  +6.01 (1.25)  single  +6.71 (1.12)
   k4k/16k q128/896                   measured  +7.64  split  +9.02 (0.85)  single +10.07 (0.76)
   k4k/16k q384/640                   measured  +5.30  split  +3.01 (1.76)  single  +3.36 (1.58)
   k8k/24k q256/768                   measured +10.16  split  +8.02 (1.27)  single  +8.95 (1.13)
   k0/16k q256/768                    measured  +7.65  split  +8.02 (0.95)  single  +8.95 (0.85)
   k4k/16k q512/512 (control, dW=0)   measured  -0.11  control
## H100-Qwen3-8B: c_X 1.013  c_S 5.189  c_S/c_X 5.12  single slope 1.121 (spread -10.1%..+7.6%)
   primary_geometry_one_to_multi: M0 56.76  M2n 2.53  M2s 14.18
   reverse_geometry_multi_to_one: M0 6.52  M2n 2.49  M2s 7.37
   k4k/16k q256/768                   measured  +7.60  split  +6.37 (1.19)  single  +7.05 (1.08)
   k4k/16k q128/896                   measured  +7.69  split  +9.56 (0.80)  single +10.58 (0.73)
   k4k/16k q384/640                   measured  +5.39  split  +3.19 (1.69)  single  +3.53 (1.53)
   k8k/24k q256/768                   measured +10.50  split  +8.50 (1.24)  single  +9.40 (1.12)
   k0/16k q256/768                    measured  +7.53  split  +8.50 (0.89)  single  +9.40 (0.80)
   k4k/16k q512/512 (control, dW=0)   measured  -0.02  control
## H100-Qwen2.5-1.5B-Instruct: c_X 0.273  c_S 4.111  c_S/c_X 15.04  single slope 0.302 (spread -24.7%..+31.1%)
   primary_geometry_one_to_multi: M0 18.85  M2n 1.68  M2s 8.08
   reverse_geometry_multi_to_one: M0 3.22  M2n 2.27  M2s 4.06
## H100-Qwen2.5-7B-Instruct: c_X 0.766  c_S 4.459  c_S/c_X 5.82  single slope 0.840 (spread -5.4%..+12.0%)
   primary_geometry_one_to_multi: M0 44.37  M2n 2.04  M2s 14.74
   reverse_geometry_multi_to_one: M0 4.99  M2n 1.81  M2s 10.16
## L20-Qwen3-4B-posthoc: c_X 6.331  c_S 8.810  c_S/c_X 1.39  single slope 6.497 (spread -1.9%..+2.3%)
   primary_geometry_one_to_multi: M0 339.76  M2n 3.07  M2s 19.86
   reverse_geometry_multi_to_one: M0 36.76  M2n 2.52  M2s 30.34
   k4k/16k q256/768                   measured +36.48  split +39.83 (0.92)  single +40.88 (0.89)
   k4k/16k q128/896                   measured +54.21  split +59.75 (0.91)  single +61.32 (0.88)
   k4k/16k q384/640                   measured +18.58  split +19.92 (0.93)  single +20.44 (0.91)
   k8k/24k q256/768                   measured +47.10  split +53.11 (0.89)  single +54.50 (0.86)
   k0/16k q256/768                    measured +54.87  split +53.11 (1.03)  single +54.50 (1.01)
   k4k/16k q512/512 (control, dW=0)   measured  -0.09  control
## A100-Qwen3-4B-posthoc: c_X 3.229  c_S 5.695  c_S/c_X 1.76  single slope 3.300 (spread -4.3%..+5.2%)
   primary_geometry_one_to_multi: M0 171.45  M2n 1.92  M2s 9.14
   reverse_geometry_multi_to_one: M0 18.34  M2n 2.31  M2s 12.69
   k4k/16k q256/768                   measured +13.55  split +20.32 (0.67)  single +20.76 (0.65)
   k4k/16k q128/896                   measured +27.62  split +30.48 (0.91)  single +31.14 (0.89)
   k4k/16k q384/640                   measured  +6.97  split +10.16 (0.69)  single +10.38 (0.67)
   k8k/24k q256/768                   measured +19.60  split +27.09 (0.72)  single +27.68 (0.71)
   k0/16k q256/768                    measured +20.78  split +27.09 (0.77)  single +27.68 (0.75)
   k4k/16k q512/512 (control, dW=0)   measured  +0.01  control
## A100-Qwen3-8B-posthoc: c_X 3.257  c_S 6.153  c_S/c_X 1.89  single slope 3.316 (spread -3.4%..+3.4%)
   primary_geometry_one_to_multi: M0 169.98  M2n 1.66  M2s 11.84
   reverse_geometry_multi_to_one: M0 18.60  M2n 1.72  M2s 22.69

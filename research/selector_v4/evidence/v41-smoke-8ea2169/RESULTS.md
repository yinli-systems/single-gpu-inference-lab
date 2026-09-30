# Selector v4.1 dual-GPU isolated-symbol smoke

Source commit: `8ea2169dbc2967e02f12e527ea0b304a264df1ab`.

- RTX4090 job1643827 and RTX5090 job1643828 both completed0:0.
- Ragged and paged pristine/native-before/cap/native-after output and LSE hashes are exact.
- Plan policy sequence is native0, cap1, native0; plan cores match pristine.
- Compiled private-JIT modules contain co-resident ragged/paged native and resource symbols on both GPUs.
- Native-after/native-before paged ratio:4090 1.000251x,5090 0.999792x; 5090 ragged1.000398x. 4090 ragged1.045457x is an order/warmup-direction diagnostic and not release evidence.
- Cap is beneficial on this exposed4090 smoke geometry (~1.36x) and regressive on5090 (~0.94x), which demonstrates why tactics must be GPU/execution/operation specific and fail closed.
- Smoke authorizes only exposed dev qualification. Canary, default and serving promotion remain OFF.

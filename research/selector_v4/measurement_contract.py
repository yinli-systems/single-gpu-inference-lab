from __future__ import annotations
import math
REVISION='4.0'
EAGER_MIN_WINDOW_US=12000.0
EAGER_TARGET_US=48000.0

def choose_eager_calls(pilots, pilot_calls=16, max_calls=4096):
    if not pilots or any(not math.isfinite(x) or x<=0 for x in pilots):raise ValueError('pilots')
    needed=max(16,math.ceil(EAGER_TARGET_US/(min(pilots)/pilot_calls)))
    count=1<<(needed-1).bit_length()
    if count>max_calls:raise ValueError('call budget')
    return count

def check_window(elapsed_us,calls):
    if not math.isfinite(elapsed_us) or elapsed_us<EAGER_MIN_WINDOW_US or calls<16:raise ValueError('short eager window')

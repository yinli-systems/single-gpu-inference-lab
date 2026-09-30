"""Pure CPU contracts for the 3.2.1 measurement revision."""
import math

REVISION = "3.2.1"
EAGER_MIN_WINDOW_US = 12000.0
EAGER_CALIBRATION_TARGET_US = 48000.0

class EagerWindowTooShort(RuntimeError):
    def __init__(self, elapsed_us, calls):
        self.details = {"elapsed_us": elapsed_us, "calls": calls,
                        "minimum_us": EAGER_MIN_WINDOW_US}
        super().__init__("Scored eager window shorter than frozen minimum: " + str(self.details))

def choose_eager_calls(pilot_window_us, pilot_calls=16, max_calls=4096):
    if not pilot_window_us or pilot_calls <= 0 or max_calls < 16:
        raise ValueError("invalid calibration")
    if any(not math.isfinite(x) or x <= 0 for x in pilot_window_us):
        raise ValueError("invalid pilot duration")
    per_call = min(pilot_window_us) / pilot_calls
    needed = max(16, math.ceil(EAGER_CALIBRATION_TARGET_US / per_call))
    count = 1 << (needed - 1).bit_length()
    if count > max_calls:
        raise ValueError("calibration exceeds frozen call budget")
    return count

def check_eager_window(elapsed_us, calls):
    if not math.isfinite(elapsed_us) or elapsed_us <= 0 or type(calls) is not int or calls < 16:
        raise ValueError("invalid eager window")
    if elapsed_us < EAGER_MIN_WINDOW_US:
        raise EagerWindowTooShort(elapsed_us, calls)

def expected_sequence(mode, group, block):
    if mode == "pristine":
        if group != "position": raise ValueError("pristine comparison group")
        return [("pristine", role) for role in (("a","b","b","a") if block%2==0 else ("b","a","a","b"))]
    if mode != "paired" or group not in ("cap", "guarded"):
        raise ValueError("paired comparison group")
    arms = ("off",group,group,"off") if block%2==0 else (group,"off","off",group)
    return [(arm, "A" if arm == "off" else "B") for arm in arms]

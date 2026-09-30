from __future__ import annotations
import unittest
from research.selector_v4.measure_v4 import _power_two, operation_identity

class MeasurementIdentityTests(unittest.TestCase):
    def test_power_two_window_rounding(self):
        self.assertEqual(_power_two(1),1);self.assertEqual(_power_two(17),32);self.assertEqual(_power_two(64),64)
    def test_off_and_cap_share_operation_key(self):
        env={'gpu_name':'NVIDIA GeForce RTX 5090','gpu_uuid':'GPU-x','num_sms':170,'driver':'580.82.07','cuda':'13.0','torch':'2.13','flashinfer':'0.7.0','nvcc':'13.0','backend_source_sha256':'a'*64,'official_overlay_sha256':'b'*64,'max_smem_per_sm':102400,'max_smem_per_block_optin':101376}
        case={'q':[3,35,99,163,259],'cached':[32768,16384,8192,2048,64]};windows={'graph16_replay':{'replays':4}}
        core=[40]+[0]*14
        off=operation_identity(env,case,'float16','paged','unsplit',core+[0],'graph16_replay',windows)
        cap=operation_identity(env,case,'float16','paged','unsplit',core+[1],'graph16_replay',windows)
        self.assertEqual(off.key,cap.key);self.assertEqual(len(off.operation['plan_signature']),15);self.assertEqual(off.operation["window_config"],{"replays":4})

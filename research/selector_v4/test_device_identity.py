import unittest
from types import SimpleNamespace
from .device_identity import device_uuid

class DeviceIdentityTests(unittest.TestCase):
    def fake(self,value):
        return SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:0,get_device_properties=lambda _:SimpleNamespace(uuid=value)))
    def test_actual_uuid_not_visible_ordinal(self):
        value='01234567-89ab-cdef-0123-456789abcdef'
        self.assertEqual(device_uuid(self.fake(value)),'GPU-'+value)
        self.assertEqual(device_uuid(self.fake('GPU-'+value)),'GPU-'+value)
    def test_unknown_device_fails_closed(self):
        for value in (None,'','0','GPU-x'):
            with self.assertRaises(RuntimeError):device_uuid(self.fake(value))

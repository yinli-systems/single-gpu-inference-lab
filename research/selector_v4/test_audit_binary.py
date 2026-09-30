import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.selector_v4.audit_binary import run

class AuditBinaryTests(unittest.TestCase):
 def fixture(self,resource=True):
  tmp=tempfile.TemporaryDirectory();root=Path(tmp.name);base=root/'.cache/flashinfer/x';base.mkdir(parents=True);header=root/'overlay/flashinfer/data/include/flashinfer/attention/prefill.cuh';header.parent.mkdir(parents=True);header.write_text('BatchPrefillWithRaggedKVCacheKernel BatchPrefillWithPagedKVCacheKernel '+('BatchPrefillWithRaggedKVCacheResourceKernel BatchPrefillWithPagedKVCacheResourceKernel' if resource else ''))
  so=base/'module.so';so.write_bytes(b'ELF')
  source='BatchPrefillWithRaggedKVCacheKernel BatchPrefillWithPagedKVCacheKernel '
  if resource:source+='BatchPrefillWithRaggedKVCacheResourceKernel BatchPrefillWithPagedKVCacheResourceKernel'
  return tmp,root
 def test_separate_symbols_required(self):
  tmp,root=self.fixture();self.addCleanup(tmp.cleanup);out=root/'audit.json'
  nm='BatchPrefillWithRaggedKVCacheKernel\nBatchPrefillWithPagedKVCacheKernel\nBatchPrefillWithRaggedKVCacheResourceKernel\nBatchPrefillWithPagedKVCacheResourceKernel\n'
  with patch('research.selector_v4.audit_binary.subprocess.run',return_value=SimpleNamespace(stdout=nm,stderr='',returncode=0)):
   run(SimpleNamespace(workspace=root,overlay=root/'overlay',out=out))
  data=json.loads(out.read_text());self.assertTrue(data['kernel_symbol_isolation_compiled']);self.assertEqual(data['same_module_pairs'],{'ragged':True,'paged':True})
 def test_missing_resource_symbol_fails(self):
  tmp,root=self.fixture(resource=False);self.addCleanup(tmp.cleanup)
  with patch('research.selector_v4.audit_binary.subprocess.run',return_value=SimpleNamespace(stdout='',stderr='',returncode=0)),self.assertRaisesRegex(RuntimeError,'generated source'):
   run(SimpleNamespace(workspace=root,overlay=root/'overlay',out=root/'audit.json'))

if __name__=='__main__':unittest.main()

import unittest
from diagnose import anomaly,signature,normalized_graph,graph_data_compat

class DiagnosisTests(unittest.TestCase):
 def test_gap_does_not_claim_cause(self):
  row=anomaly({'wall_us':1289.334125,'device_us':1196.938038})
  self.assertTrue(row['host_span_separation']);self.assertFalse(row['cause_proven'])
 def test_wall_only_has_no_invented_device_span(self):
  self.assertIsNone(anomaly({'wall_us':1200.,'device_us':None})['wall_minus_device_us'])
 def test_near_equal_does_not_flag(self):
  self.assertFalse(anomaly({'wall_us':1200.,'device_us':1199.})['host_span_separation'])
 def test_graph_signature_ignores_instance_id_not_launch(self):
  a={'nodes':[{'index':0,'node_type':'kernel','kernel_name':'x','grid_dim':[42,1,8],'block_dim':[32,4,1],'shared_mem_bytes':49152,'dependencies':[],'dependents':[],'graph_id':1}]}
  b={'nodes':[dict(a['nodes'][0],graph_id=2)]}
  self.assertEqual(signature(a),signature(b))
  b['nodes'][0]['shared_mem_bytes']=65536;self.assertNotEqual(signature(a),signature(b))

class CompatGraphDataTests(unittest.TestCase):
 def test_cuda13_0_topology_without_tools_id(self):
  class NT:
   cudaGraphNodeTypeKernel=0;cudaGraphNodeTypeMemcpy=1;cudaGraphNodeTypeMemset=2;cudaGraphNodeTypeHost=3;cudaGraphNodeTypeGraph=4;cudaGraphNodeTypeEmpty=5;cudaGraphNodeTypeWaitEvent=6;cudaGraphNodeTypeEventRecord=7;cudaGraphNodeTypeMemAlloc=8;cudaGraphNodeTypeMemFree=9
  class RT:
   cudaGraphNodeType=NT
   @staticmethod
   def cudaGraphGetNodes(graph,numNodes=0):return (0,None,2) if numNodes==0 else (0,[101,102],2)
   @staticmethod
   def cudaGraphNodeGetType(node):return (0,0 if node==101 else 5)
   @staticmethod
   def cudaGraphGetEdges(graph,numEdges=0):return (0,None,None,None,1) if numEdges==0 else (0,[101],[102],[None],1)
   @staticmethod
   def cudaGraphGetId(graph):return (0,77)
  class Params:
   gridDimX=42;gridDimY=1;gridDimZ=8;blockDimX=32;blockDimY=4;blockDimZ=1;sharedMemBytes=49152;func=555
  class Result:CUDA_SUCCESS=0
  class DRV:
   CUresult=Result
   @staticmethod
   def CUgraphNode(init_value):return init_value
   @staticmethod
   def CUfunction(init_value):return init_value
   @staticmethod
   def cuGraphKernelNodeGetParams(node):return (0,Params())
   @staticmethod
   def cuFuncGetName(func):return (0,b'kernel_x')
  class G:
   def raw_cuda_graph(self):return 999
  meta=graph_data_compat(G(),RT,DRV)
  self.assertFalse(meta['tools_id_available']);self.assertTrue(meta['graph_id_available']);self.assertEqual(meta['graph_id'],77);self.assertEqual(len(meta['nodes']),2)
  k=meta['nodes'][0];self.assertEqual(k['kernel_name'],'kernel_x');self.assertEqual(k['grid_dim'],[42,1,8]);self.assertEqual(k['block_dim'],[32,4,1]);self.assertEqual(k['shared_mem_bytes'],49152);self.assertEqual(k['dependents'],[1]);self.assertEqual(meta['nodes'][1]['dependencies'],[0])

 def test_graph_id_not_supported_is_nonfatal(self):
  class NT:
   cudaGraphNodeTypeKernel=0;cudaGraphNodeTypeMemcpy=1;cudaGraphNodeTypeMemset=2;cudaGraphNodeTypeHost=3;cudaGraphNodeTypeGraph=4;cudaGraphNodeTypeEmpty=5;cudaGraphNodeTypeWaitEvent=6;cudaGraphNodeTypeEventRecord=7;cudaGraphNodeTypeMemAlloc=8;cudaGraphNodeTypeMemFree=9
  class RT:
   cudaGraphNodeType=NT
   @staticmethod
   def cudaGraphGetNodes(graph,numNodes=0):return (0,None,1) if numNodes==0 else (0,[101],1)
   @staticmethod
   def cudaGraphNodeGetType(node):return (0,5)
   @staticmethod
   def cudaGraphGetEdges(graph,numEdges=0):return (0,None,None,None,0)
   @staticmethod
   def cudaGraphGetId(graph):return (36,None)
  class Result:CUDA_SUCCESS=0
  class DRV:
   CUresult=Result
   @staticmethod
   def CUgraphNode(init_value):return init_value
   @staticmethod
   def CUfunction(init_value):return init_value
   @staticmethod
   def cuGraphKernelNodeGetParams(node):raise AssertionError('empty node')
   @staticmethod
   def cuFuncGetName(func):raise AssertionError('empty node')
  class G:
   def raw_cuda_graph(self):return 999
  meta=graph_data_compat(G(),RT,DRV);self.assertIsNone(meta['graph_id']);self.assertFalse(meta['graph_id_available']);self.assertEqual(len(meta['nodes']),1)
 def test_signature_is_stable_without_graph_id(self):
  m={'graph_id':1,'tools_id_available':False,'nodes':[{'index':0,'node_type':'kernel','kernel_name':'x','grid_dim':[1,1,1],'block_dim':[32,1,1],'shared_mem_bytes':0,'dependencies':[],'dependents':[]}]}
  n=dict(m,graph_id=2);self.assertEqual(signature(m),signature(n))

class FastRejectTests(unittest.TestCase):
 def test_equivalence_for_all_boolean_boundaries(self):
  from fast_reject_patch import equivalence_witness
  import itertools
  for pairing,split,descriptor,batch in itertools.product((False,True),(False,True),(False,True),range(10)):
   self.assertTrue(equivalence_witness(pairing,split,descriptor,batch))
 def test_source_rejects_unknown_anchor(self):
  from fast_reject_patch import apply
  with self.assertRaises(ValueError):apply('wrong source')
 def test_skips_only_logically_false_region(self):
  from fast_reject_patch import apply
  text='  } else {\n    // Selector v3: unchanged predicate'
  patched=apply(text);self.assertIn('!plan_info.split_kv && batch_size >= 5',patched)
if __name__=='__main__':unittest.main()

"""Synthetic bookkeeping fixtures only; not model or GPU results."""
import copy
import hashlib
import json
import unittest
from audit_observations import audit_workload, percentiles


def fixture():
    cells = [dict(id='r', prompt=[101,102], max_tokens=3, arrival=0.)]
    rec = dict(id='r', prompt_tokens=2, expected_output_tokens=3, admission=.1,
               scheduled_arrival=0., token_ids=[1,2,3], token_times=[.2,.23,.26],
               finished=True, ttft=.1, offered_ttft=.2, latency=.16,
               offered_latency=.26, tpot=.03, strict_slo=True, lenient_slo=True)
    payload = dict(complete=True, elapsed_seconds=.3, requests=1, output_tokens=3,
                   tokens_per_second=10., requests_per_second=1/.3,
                   strict_slo_count=1, lenient_slo_count=1,
                   strict_slo_goodput=1/.3, lenient_slo_goodput=1/.3,
                   requests_detail=[rec], router={'counts':{'qualified_plans':1}},
                   workload_sha256=hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest())
    for k,v in [('ttft_seconds',.1),('offered_ttft_seconds',.2),('tpot_seconds',.03),
                ('request_latency_seconds',.16),('offered_latency_seconds',.26)]:
        payload[k]=dict(p50=v,p95=v,p99=v)
    steps=[dict(started=t-.01,completed=t,outputs=[dict(id='r',added=1,finished=(i==2))])
           for i,t in enumerate((.2,.23,.26))]
    return payload,steps,cells


class ObservationAuditTests(unittest.TestCase):
    def test_valid_raw_trace(self):
        p,s,c=fixture();r=audit_workload(p,s,c)
        self.assertTrue(r['passed']);self.assertEqual(r['tokens'],3)
    def test_missing_final_step(self):
        p,s,c=fixture()
        with self.assertRaises(ValueError):audit_workload(p,s[:-1],c)
    def test_step_clock_overlap(self):
        p,s,c=fixture();s[1]['started']=.19
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_step_clock_nan(self):
        p,s,c=fixture();s[0]['completed']=float('nan')
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_latency_summary_tampering(self):
        p,s,c=fixture();p['ttft_seconds']['p95']=0.
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_throughput_tampering(self):
        p,s,c=fixture();p['tokens_per_second']=100.
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_record_time_tampering(self):
        p,s,c=fixture();p['requests_detail'][0]['token_times'][0]=.18
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_false_slo(self):
        p,s,c=fixture();p['requests_detail'][0]['strict_slo']=False
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_false_finished(self):
        p,s,c=fixture();p['requests_detail'][0]['finished']=False
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_wrong_prompt_length(self):
        p,s,c=fixture();p['requests_detail'][0]['prompt_tokens']=3
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_wrong_output_length(self):
        p,s,c=fixture();p['requests_detail'][0]['expected_output_tokens']=4
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_duplicate_request(self):
        p,s,c=fixture();p['requests_detail'].append(copy.deepcopy(p['requests_detail'][0]))
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_unknown_step_request(self):
        p,s,c=fixture();s[0]['outputs'][0]['id']='other'
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_repeated_finished_event(self):
        p,s,c=fixture();s.append(dict(started=.28,completed=.29,outputs=[dict(id='r',added=0,finished=True)]))
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_invalid_token_id(self):
        p,s,c=fixture();p['requests_detail'][0]['token_ids'][0]=-1
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_boolean_event_count(self):
        p,s,c=fixture();s[0]['outputs'][0]['added']=True
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_count_trace_mismatch(self):
        p,s,c=fixture();s[0]['outputs'][0]['added']=0
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_output_before_admission(self):
        p,s,c=fixture();p['requests_detail'][0]['admission']=.25
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_finished_after_wallclock(self):
        p,s,c=fixture();p['elapsed_seconds']=.24
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_changed_workload_hash(self):
        p,s,c=fixture();p['workload_sha256']='0'*64
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_empty_workload(self):
        p,s,c=fixture()
        with self.assertRaises(ValueError):audit_workload(p,[],[])
    def test_no_adapter_execution(self):
        p,s,c=fixture();p['router']['counts']['qualified_plans']=0
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_mislabeled_arrival(self):
        p,s,c=fixture();p['requests_detail'][0]['scheduled_arrival']=.1
        with self.assertRaises(ValueError):audit_workload(p,s,c)
    def test_empty_quantiles(self):
        with self.assertRaises(ValueError):percentiles([])
    def test_nonfinite_quantiles(self):
        with self.assertRaises(ValueError):percentiles([float('inf')])
    def test_boolean_quantiles(self):
        with self.assertRaises(ValueError):percentiles([True])


if __name__=='__main__':unittest.main()

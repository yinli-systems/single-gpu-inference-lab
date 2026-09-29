"""Lossless bounded SSE diagnostics. Incomplete streams remain failures with evidence."""
from __future__ import annotations
import json,time

class StreamState:
    def __init__(self, expected):
        if type(expected) is not int or expected<=0:raise ValueError('invalid expected length')
        self.expected=expected;self.tokens=[];self.count=0;self.finish=None
        self.done=False;self.frames=[];self.error=None
    def accept(self, data):
        if len(self.frames)>=1024:raise ValueError('frame bound exceeded')
        self.frames.append(data)
        if data=='[DONE]':self.done=True;return
        if self.done:raise ValueError('frame after DONE')
        obj=json.loads(data)
        if 'error' in obj:raise ValueError('server error: '+str(obj['error']))
        meta=obj.get('meta_info',{});n=meta.get('completion_tokens');ids=obj.get('output_ids',[])
        if type(n) is not int or n<self.count or n>self.expected:raise ValueError('completion count invalid')
        if not isinstance(ids,list) or any(type(x) is not int or x<0 for x in ids):raise ValueError('invalid output IDs')
        if len(ids)==n-self.count:new=ids
        elif len(ids)==n and ids[:self.count]==self.tokens:new=ids[self.count:]
        elif n==self.count and not ids:new=[]
        else:raise ValueError('token/prefix length invalid')
        self.tokens+=new;self.count=n
        if meta.get('finish_reason') is not None:self.finish=meta['finish_reason']
    def result(self):
        complete=self.error is None and self.done and len(self.tokens)==self.expected and self.finish is not None
        return dict(complete=complete,expected_tokens=self.expected,received_tokens=len(self.tokens),
            tokens=self.tokens,finish_reason=self.finish,done_marker=self.done,frames=self.frames,error=self.error)

async def record_request(session,url,cell,wire_id):
    state=StreamState(cell['output_tokens']);started=time.perf_counter();status=None;headers={}
    payload=dict(rid=wire_id,input_ids=cell['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=cell['output_tokens'],ignore_eos=True),stream=True)
    try:
        async with session.post(url+'/generate',json=payload) as response:
            status=response.status;headers=dict(response.headers)
            if status!=200:
                state.error='HTTP '+str(status)+': '+(await response.text())[:4000]
            else:
                # SGLang emits one data field per event; retain comments and final EOF state.
                buffer=[]
                async for line in response.content:
                    text=line.decode('utf-8').rstrip('\r\n')
                    if text=='':
                        if buffer:state.accept('\n'.join(buffer));buffer=[]
                    elif text.startswith('data:'):buffer.append(text[5:].lstrip(' '))
                if buffer:state.accept('\n'.join(buffer))
    except Exception as exc:state.error=type(exc).__name__+': '+str(exc)
    return dict(logical_id=cell['id'],wire_id=wire_id,status=status,headers=headers,
        elapsed_seconds=time.perf_counter()-started,diagnostic_only=True,**state.result())

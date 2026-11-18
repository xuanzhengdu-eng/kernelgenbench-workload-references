import pickle
from types import SimpleNamespace
import pytest
import datetime,math,os,tempfile,time,torch
import torch.distributed as dist,torch.multiprocessing as mp
from sandbox.verifier.test_parametrize import label,parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from kernelgenbench.dataset.baseline.v1_synthetic.v1_megatron_row_nvls_fused_ import v1_megatron_row_nvls_fused_ as ref
# TP=2, (sequence,batch,input features per rank, output features); all BF16.
COMBOS = [
    # (sequence, batch, input features per rank, output features); all bfloat16.
    (16, 1, 16, 32),
    (16, 2, 32, 32),
    (16, 1, 32, 64),
]
def get_input(case,rank,device='cpu'):
 s,b,i,o=case;g=torch.Generator().manual_seed(1000+rank*23+s+b+i+o)
 def rnd(*shape):return (torch.randn(*shape,generator=g)*.2).bfloat16().to(device)
 return dict(x=rnd(s,b,i),weight=rnd(o,i),residual=rnd(s//2,b,o),symmetric_buffer=torch.full((s,b,o),99,dtype=torch.bfloat16,device=device),gamma=(1+rnd(o).float()).bfloat16(),eps=1e-5)
def input_build(case,rank):return get_input(case,rank)
def compute_res(**kw):
 import kernelgenbench
 return kernelgenbench.triton.v1_megatron_row_nvls_fused_(**kw)
def oracle(case,rank):
 s,b,i,o=case;data=[get_input(case,r) for r in range(2)]
 partial=[(d['x'].double()@d['weight'].double().t()).bfloat16().float() for d in data]
 summed=(partial[0]+partial[1]).bfloat16();chunks=[];rout=[]
 for r in range(2):
  res=(summed[r*s//2:(r+1)*s//2].float()+data[r]['residual'].float()).bfloat16();rout.append(res)
  z=res.double();w=data[r]['gamma'].double();chunks.append((z*torch.rsqrt(z.square().mean(-1,keepdim=True)+data[r]['eps'])*w).bfloat16())
 return rout[rank],torch.cat(chunks,0)
def worker(rank,case,path,outdir,candidate_bytes):
 import kernelgenbench
 kernelgenbench.triton = SimpleNamespace(v1_megatron_row_nvls_fused_=pickle.loads(candidate_bytes))
 torch.cuda.set_device(rank)
 torch.set_num_threads(1);dist.init_process_group('nccl',init_method='file://'+path,rank=rank,world_size=2,device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=60))
 try:
  a,b=get_input(case,rank,rank),get_input(case,rank,rank);saved={k:v.clone() for k,v in b.items() if isinstance(v,torch.Tensor) and k not in ('residual','symmetric_buffer')};ptr=b['residual'].data_ptr();bp=b['symmetric_buffer'].data_ptr()
  ref(**a);out=compute_res(**b);ex,buf=(t.to(rank) for t in oracle(case,rank))
  assert out is b['residual'] and out.data_ptr()==ptr and b['symmetric_buffer'].data_ptr()==bp
  for d in (a,b):torch.testing.assert_close(d['residual'],ex,atol=.01,rtol=.01);torch.testing.assert_close(d['symmetric_buffer'],buf,atol=.02,rtol=.02)
  for k,v in saved.items():assert torch.equal(v,b[k])
  def bench(fn):
   items=[get_input(case,rank,rank) for _ in range(2)];dist.barrier();start=time.perf_counter()
   for d in items:fn(**d)
   dist.barrier();return (time.perf_counter()-start)*500
  ta,tb=bench(ref),bench(compute_res)
  import json
  with open(os.path.join(outdir,str(rank)+'.json'),'w') as f:json.dump([ta,tb],f)
 finally:dist.destroy_process_group()
@label('v1_megatron_row_nvls_fused_')
@parametrize('case',COMBOS)
@pytest.mark.parametrize('case',COMBOS)
def test_accuracy_v1_megatron_row_nvls_fused_(case):
 import json, kernelgenbench
 candidate = kernelgenbench.triton.v1_megatron_row_nvls_fused_
 try:
  encoded = pickle.dumps(candidate)
 except (AttributeError,pickle.PicklingError):
  import cloudpickle
  encoded = cloudpickle.dumps(candidate)
 with tempfile.TemporaryDirectory(prefix='row-nvls-fused-') as tmp:
  mp.start_processes(worker,args=(case,os.path.join(tmp,'store'),tmp,encoded),nprocs=2,join=True,start_method='spawn');timings=[json.load(open(os.path.join(tmp,str(r)+'.json'))) for r in range(2)]
 ta=max(x[0] for x in timings);tb=max(x[1] for x in timings)
 return CustomBenchmarkResult(ref_time=ta,res_time=tb,speedup=ta/tb)

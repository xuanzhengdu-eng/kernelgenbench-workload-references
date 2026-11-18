"""CPU Lloyd-Max Gaussian quantizer initialization, finite trapezoid integration.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/turboquant/centroids.py#L18-L84
"""
import math
import torch

def v1_vllm_lloyd_max(d,bits,max_iter=200,tol=1e-10):
    n=2**bits;sigma=math.sqrt(1.0/d);lo=-3.5*sigma;hi=3.5*sigma
    c=torch.tensor([lo+(hi-lo)*(i+.5)/n for i in range(n)],dtype=torch.float64)
    t=torch.linspace(0.,1.,201,dtype=torch.float64);w=torch.ones(201,dtype=torch.float64);w[0]=w[-1]=.5
    for _ in range(max_iter):
        bounds=(c[:-1]+c[1:])*.5;edges=torch.cat((torch.tensor([lo*3],dtype=c.dtype),bounds,torch.tensor([hi*3],dtype=c.dtype)))
        x=edges[:-1,None]+(edges[1:]-edges[:-1])[:,None]*t
        pdf=torch.exp(-x*x/(2*sigma*sigma))/(math.sqrt(2*math.pi)*sigma)
        step=(edges[1:]-edges[:-1])/200.;den=(pdf*w).sum(dim=1)*step;num=(x*pdf*w).sum(dim=1)*step
        new=torch.where(den>1e-15,num/den,c)
        # The source breaks before assigning new on the converged iteration.
        if float((new-c).abs().max())<tol:break
        c=new
    return c.float(),((c[:-1]+c[1:])*.5).float()

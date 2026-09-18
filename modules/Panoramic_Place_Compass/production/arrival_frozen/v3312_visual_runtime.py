from __future__ import annotations
import hashlib, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch

ROOT=Path(os.environ.get('PIVOTNAV_ARRIVAL_DEPENDENCY_ROOT', str(Path(__file__).resolve().parent)))
V3310=Path(os.environ.get(
 'V3313_VISUAL_DEPENDENCY_ROOT',
 os.environ.get('PIVOTNAV_ARRIVAL_DEPENDENCY_ROOT', str(ROOT)),
))

def tensor_rgb(rgb):
 value=cv2.resize(np.asarray(rgb)[...,:3],(448,224),interpolation=cv2.INTER_AREA)
 return torch.from_numpy(np.ascontiguousarray(value)).permute(2,0,1)

class R361:
 def __init__(self,device):
  package=V3310/'frozen_inputs/r361_package'
  os.environ['PANORAMIC_VPR_DINOV2_CHECKOUT']=str(package/'third_party/dinov2')
  os.environ['PANORAMIC_VPR_DINOV2_WEIGHT']=str(package/'cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
  os.environ['TORCH_HOME']=str(package/'cache/torch')
  sys.path[:0]=[str(package/'python'),str(V3310)]
  from stable_retrieval_runtime import StableR361RetrievalRuntime
  self.device=torch.device(device); torch.cuda.set_device(self.device)
  self.adapter=StableR361RetrievalRuntime(package/'r361_modular.pt',device=device,precision='fp32'); self.runtime=self.adapter.runtime
 def encode(self,rgb):
  prepared,_=self.runtime._prepare_image(tensor_rgb(rgb))
  with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False): return self.runtime._encode_normalized(prepared.float())
 def pair(self,q,g):
  from torch.nn import functional as F
  with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
   yaw=self.runtime._track_y(q,g); private=self.runtime.arrival_feature_bearing_head(q['tokens'].float(),g['tokens'].float()); sim=F.cosine_similarity(q['global_descriptor'].float(),g['global_descriptor'].float(),dim=-1); sectors=yaw.get('descriptor_correlation_32')
   if sectors is None: sectors=yaw['scale_scores'].float().mean(dim=1)
   scalar=torch.cat((sim.unsqueeze(-1),self.runtime._correlation_summary(sectors),torch.stack((yaw['top1_probability'].float(),yaw['yaw_confidence'].float(),yaw['normalized_entropy'].float()),dim=-1),torch.stack((private['bearing_confidence'].float(),private['bearing_valid_probability'].float(),private['normalized_entropy'].float()),dim=-1),private['local_correlation_summary'].float(),torch.zeros(sim.shape[0],1,device=self.device),torch.ones(sim.shape[0],1,device=self.device)),dim=-1)
   spatial=torch.cat((yaw['pair_embedding'].float(),private['pair_embedding'].float()),dim=-1); feature=self.runtime._summarize_arrival_feature(spatial,scalar); visual=self.runtime.arrival_head.visual_encoder(feature.float()); probability=torch.sigmoid(self.runtime.arrival_head.frame_logit(visual).squeeze(-1))
  return {'arrival_probability':float(probability[0]),'yaw_degrees':float(yaw['predicted_yaw_degrees'][0]),'yaw_confidence':float(yaw['yaw_confidence'][0]),'vpr_similarity':float(sim[0])}

def coverage(points,w=256,h=256,n=6):
 return len({(min(n-1,int(x/w*n)),min(n-1,int(y/h*n))) for x,y in points})/(n*n) if len(points) else 0.0

class LightGlueGeometry:
 def __init__(self,device):
  sys.path[:0]=[str(V3310/'scripts'),str(V3310),str(V3310/'frozen_inputs/LightGlue')]
  from lightglue import LightGlue,SuperPoint
  self.device=torch.device(device); os.environ['TORCH_HOME']=str(V3310/'frozen_inputs/lightglue/cache/torch')
  self.extractor=SuperPoint(max_num_keypoints=512).eval().to(self.device)
  self.matcher=LightGlue(features='superpoint',n_layers=9,flash=True,mp=False,depth_confidence=.95,width_confidence=.99,filter_threshold=.1).eval().to(self.device)
  self.fx=128/math.tan(math.radians(50)); self.K=np.array([[self.fx,0,127.5],[0,self.fx,127.5],[0,0,1]],float)
 def analyze(self,query,goal,yaw):
  from erp_projection import PerspectiveViewSpec,project
  from lightglue.utils import numpy_image_to_torch,rbd
  sectors=[]; started=time.perf_counter()
  for sid,gc in enumerate([45.*i for i in range(8)]):
   qi,_=project(query,PerspectiveViewSpec(-1,(gc+yaw)%360,100.,90.,256,256)); gi,_=project(goal,PerspectiveViewSpec(-1,gc,100.,90.,256,256))
   with torch.inference_mode():
    a=self.extractor.extract(numpy_image_to_torch(qi).to(self.device),resize=None); b=self.extractor.extract(numpy_image_to_torch(gi).to(self.device),resize=None); m=rbd(self.matcher({'image0':a,'image1':b}))
   matches=m['matches'].detach().cpu().numpy(); ka=rbd(a)['keypoints'].detach().cpu().numpy(); kb=rbd(b)['keypoints'].detach().cpu().numpy(); p0=ka[matches[:,0]] if len(matches) else np.empty((0,2)); p1=kb[matches[:,1]] if len(matches) else np.empty((0,2))
   H=F=E=hm=fm=em=None
   if len(matches)>=4: H,hm=cv2.findHomography(p0,p1,cv2.RANSAC,3.0)
   if len(matches)>=8: F,fm=cv2.findFundamentalMat(p0,p1,cv2.FM_RANSAC,1.5,.999)
   if len(matches)>=5:
    try:E,em=cv2.findEssentialMat(p0,p1,self.K,method=cv2.RANSAC,prob=.999,threshold=1.5)
    except cv2.error:E=em=None
   mask=lambda x: np.zeros(len(matches),bool) if x is None or len(x)!=len(matches) else x.reshape(-1).astype(bool)
   hm,fm,em=mask(hm),mask(fm),mask(em); pe=p0[em]; ge=p1[em]; hull=min(cv2.contourArea(cv2.convexHull(pe.astype(np.float32))) if len(pe)>=3 else 0,cv2.contourArea(cv2.convexHull(ge.astype(np.float32))) if len(ge)>=3 else 0)/(256*256)
   pos=0.0
   if E is not None and em.sum()>=5:
    try: pos=cv2.recoverPose(np.asarray(E)[:3],p0,p1,self.K,mask=em.astype(np.uint8)[:,None])[0]/max(int(em.sum()),1)
    except cv2.error: pass
   reproj=99.
   if H is not None and len(p0):
    try: reproj=float(np.median(np.linalg.norm(cv2.perspectiveTransform(p0[:,None].astype(float),H)[:,0]-p1,axis=1)))
    except Exception: pass
   sectors.append({'raw':len(matches),'e':int(em.sum()),'er':float(em.mean()) if len(em) else 0.,'h':float(hm.mean()) if len(hm) else 0.,'f':float(fm.mean()) if len(fm) else 0.,'grid':min(coverage(pe),coverage(ge)),'hull':float(hull),'hs':min(float(np.ptp(pe[:,0]))/256 if len(pe) else 0,float(np.ptp(ge[:,0]))/256 if len(ge) else 0),'vs':min(float(np.ptp(pe[:,1]))/256 if len(pe) else 0,float(np.ptp(ge[:,1]))/256 if len(ge) else 0),'reproj':reproj,'pos':float(pos)})
  torch.cuda.synchronize(self.device); best=max(sectors,key=lambda x:x['e'])
  return {'e_inliers':best['e'],'e_inlier_ratio':best['er'],'grid_coverage':best['grid'],'hull_area':best['hull'],'horizontal_span':best['hs'],'vertical_span':best['vs'],'reprojection_error':best['reproj'],'positive_depth_ratio':best['pos'],'supported_sectors':sum(x['e']>=8 for x in sectors),'h_dominance':best['h']-max(best['er'],best['f']),'latency_ms':(time.perf_counter()-started)*1000,'sectors':sectors}

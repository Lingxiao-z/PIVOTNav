from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from modules.Panoramic_Place_Compass.arrival import BoundaryVerifier,Decision,Evidence,Thresholds,absolute_decision,direct_target_contract,geometry_strength

ROOT=Path(os.environ.get('PIVOTNAV_OMNIGUARD_VALIDATION_ROOT', Path(__file__).resolve().parents[4]))
SOURCE_OUT=ROOT/'output/omniguard_v3312_boundary_active_approach'
PANO=Path(os.environ.get('PIVOTNAV_NAVIGATION_ROOT', Path(__file__).resolve().parents[4]))
OUT=PANO/'outputs/integration_v2_behavioral/stage_i5/arrival_active_evidence_v2r2'
PROJECT=Path(os.environ.get('PIVOTNAV_HABITAT_ROOT', Path.cwd()))
OMNI=Path(os.environ.get('PIVOTNAV_OMNIGUARD_ROOT', '')).expanduser()
WEIGHT=Path(os.environ.get('PIVOTNAV_OMNIGUARD_CHECKPOINT', '')).expanduser()
WORKER=Path(os.environ.get('PIVOTNAV_OMNIGUARD_WORKER', str(ROOT/'omniguard_worker.py'))).expanduser()
PYTHON=os.environ.get('PIVOTNAV_OMNIGUARD_PYTHON', sys.executable)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def wrap(v):return (v+180)%360-180
def quat_heading(deg):
 r=math.radians(deg);return [0.0,math.sin(-r/2),0.0,math.cos(-r/2)]
def make_episode(trial,start,heading,path):
 ep=dict(trial['episode']['source_episode']);ep['start_position']=list(start);ep['start_rotation']=quat_heading(heading);g=list(trial['episode']['target_position_audit_only']);audit=list(g);audit[0]+=.1;ep['goals']=[{'position':audit,'radius':.2}];path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'wt') as f:json.dump({'episodes':[ep]},f)
def env_config(ep,gpu):
 os.chdir(PROJECT);sys.path.insert(0,str(PROJECT/'scripts_gs'));from habitat.config.default import get_config;from habitat.config.default_structured_configs import HabitatConfigPlugin,VelocityControlActionConfig,register_hydra_plugin;from habitat.config.read_write import read_write;from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin;from omegaconf import OmegaConf
 register_hydra_plugin(HabitatBaselinesConfigPlugin);register_hydra_plugin(HabitatConfigPlugin);cfg=get_config(str(PROJECT/'data/scene_datasets/gs_scenes/configs/ddppo_panoramic_rgb_imagenav_gs_eval.yaml'),overrides=['~habitat.task.measurements.top_down_map']);spec=VelocityControlActionConfig(lin_vel_range=[-0.4,.4],ang_vel_range=[-math.degrees(1),math.degrees(1)],min_abs_lin_speed=-1,min_abs_ang_speed=-1,time_step=.1)
 with read_write(cfg):
  cfg.habitat.seed=3312001;cfg.habitat.dataset.split='train';cfg.habitat.dataset.data_path=str(Path(ep).resolve());cfg.habitat.dataset.scenes_dir='data/scene_datasets';cfg.habitat.dataset.content_scenes=['*'];cfg.habitat.environment.max_episode_steps=300;cfg.habitat.simulator.scene_dataset='data/scene_datasets/gs_scenes/hm3d_annotated_basis.scene_dataset_config.json';cfg.habitat.simulator.forward_step_size=.25;cfg.habitat.simulator.turn_angle=30
  if 'gpu_device_id' in cfg.habitat.simulator:cfg.habitat.simulator.gpu_device_id=gpu
  elif 'habitat_sim_v0' in cfg.habitat.simulator and 'gpu_device_id' in cfg.habitat.simulator.habitat_sim_v0:cfg.habitat.simulator.habitat_sim_v0.gpu_device_id=gpu
  sensors=cfg.habitat.simulator.agents.main_agent.sim_sensors;sensors.clear();sensors.rgb_sensor={'type':'HabitatSimEquirectangularRGBSensor','height':256,'width':512,'position':[0.,1.5,0.],'orientation':[0.,0.,0.]};cfg.habitat.task.actions.velocity_control=OmegaConf.structured(spec)
 return cfg
def vel(v,w):return {'action':'velocity_control','action_args':{'linear_velocity':float(np.clip(v/.4,-1,1)),'angular_velocity':float(np.clip(-w,-1,1)),'time_step':.1,'allow_sliding':False}}
def state(env):
 from habitat_sim.utils.common import quat_rotate_vector
 s=env.sim.get_agent_state();f=np.asarray(quat_rotate_vector(s.rotation,np.array([0.,0.,-1.])));h=math.degrees(math.atan2(float(f[0]),float(-f[2])));return np.asarray(s.position,dtype=float),h
class Worker:
 def __init__(self,cmd,err):
  err.parent.mkdir(parents=True,exist_ok=True);self.e=open(err,'w');self.p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.e,text=True,bufsize=1);self.ready=json.loads(self.p.stdout.readline())
 def ask(self,x):self.p.stdin.write(json.dumps(x)+'\n');self.p.stdin.flush();return json.loads(self.p.stdout.readline())
 def close(self):
  try:self.ask({'command':'close'})
  except:pass
  self.p.terminate();self.e.close()
def evidence(runtime,geom,margin,lgmargin,stable,consistency):
 vals={'arrival_probability':runtime['arrival_probability'],'vpr_similarity':runtime['vpr_similarity'],'candidate_margin':margin,'lightglue_second_margin':lgmargin,'e_inliers':geom['e_inliers'],'e_inlier_ratio':geom['e_inlier_ratio'],'grid_coverage':geom['grid_coverage'],'hull_area':geom['hull_area'],'horizontal_span':geom['horizontal_span'],'vertical_span':geom['vertical_span'],'reprojection_error':geom['reprojection_error'],'positive_depth_ratio':geom['positive_depth_ratio'],'supported_sectors':geom['supported_sectors'],'h_dominance':geom['h_dominance'],'yaw_consistent':math.isfinite(wrap(runtime['yaw_degrees'])),'candidate_stable':stable,'multiview_consistency':consistency,'repeated_texture_risk':geom['grid_coverage']<.02 and geom['e_inliers']>=12,'through_wall_risk':False,'vggt_veto':False}
 return Evidence.from_runtime(vals),vals
def pair_from_encoding(runtime,q,g):
 from torch.nn import functional as F
 with torch.inference_mode():
  pair=runtime._pair_outputs(q,g);feature=pair['arrival_feature'].float().unsqueeze(1);visual=runtime.arrival_head.visual_encoder(feature);probability=torch.sigmoid(runtime.arrival_head.frame_logit(visual).squeeze(-1))[:,0];similarity=F.cosine_similarity(q['global_descriptor'].float(),g['global_descriptor'].float(),dim=-1);yaw=pair['yaw']
 return {'arrival_probability':float(probability[0]),'vpr_similarity':float(similarity[0]),'yaw_degrees':float(yaw['predicted_yaw_degrees'][0]),'yaw_confidence':float(yaw['yaw_confidence'][0])}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--limit',type=int,default=48);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--run-id',required=True);a=ap.parse_args()
 if not OMNI.is_dir() or not WEIGHT.is_file() or not WORKER.is_file():
  raise RuntimeError('set PIVOTNAV_OMNIGUARD_ROOT, PIVOTNAV_OMNIGUARD_CHECKPOINT, and PIVOTNAV_OMNIGUARD_WORKER before active-evidence collection')
 run=OUT/'shards'/a.run_id
 if run.exists():raise SystemExit(f'refuse overwrite {run}')
 run.mkdir(parents=True);trials=[json.loads(x) for x in open(SOURCE_OUT/'effective_development_trial_manifest.jsonl') if x.strip()];subset=trials[a.start:a.start+a.limit]
 collections={}
 registry=json.load(open(SOURCE_OUT/'effective_collection_registry.json'))
 for x in registry['records']:collections[x['trial_id']]=json.load(open(x['trajectory_audit']))
 missing=[t['trial_id'] for t in subset if t['trial_id'] not in collections]
 if missing:raise RuntimeError(f'missing collections {missing[:5]} count={len(missing)}')
 sys.path[:0]=[str(PANO),str(PANO/'models/arrival_verifier_frozen')]
 from modules.Panoramic_Place_Compass.localization import R361Adapter
 from modules.Panoramic_Place_Compass.geometry import load_lightglue_geometry
 r=R361Adapter(PANO/'models/r361',device='cpu');lg=load_lightglue_geometry(f'cuda:{a.gpu}')
 targets=[];target_enc=[]
 for t in trials:
  audit=collections.get(t['trial_id'])
  if audit is None:
   # Cross-shard galleries are diagnostic only; use target refs available in all completed collections.
   continue
  im=cv2.cvtColor(cv2.imread(audit['target_reference']['path']),cv2.COLOR_BGR2RGB);targets.append((t['trial_id'],im,audit['target_reference']['path']));target_enc.append(r.encode_panorama(im))
 if not target_enc:
  raise RuntimeError('no collected target references available for active verification')
 gallery=torch.nn.functional.normalize(torch.cat([x['global_descriptor'].float() for x in target_enc]),dim=-1);gindex={x[0]:i for i,x in enumerate(targets)}
 omni=Worker([PYTHON,str(WORKER),'--repo',str(OMNI),'--checkpoint',str(WEIGHT),'--device',f'cuda:{a.gpu}','--output-dir',str(run/'omniguard'),'--profile','short_edge_v1'],run/'omniguard.stderr.log')
 import habitat
 results=[];steps=[];lat=[];contract=direct_target_contract()
 try:
  for n,t in enumerate(subset):
   audit=collections[t['trial_id']];last=audit['navigation_actions'][-1];ep=OUT/'episode_manifests'/a.run_id/f'{t["trial_id"]}.json.gz';make_episode(t,last['position_gt_audit_only'],last['heading_gt_audit_only'],ep);env=habitat.Env(config=env_config(ep,a.gpu));root=run/t['trial_id'];root.mkdir()
   try:
    obs=env.reset();query=np.asarray(obs['rgb'])[...,:3].copy();goal=targets[gindex[t['trial_id']]][1];goalenc=target_enc[gindex[t['trial_id']]];history=[];views=[];started=time.perf_counter();omni.ask({'command':'reset'});p0,h0=state(env);target=np.asarray(t['episode']['target_position_audit_only'])
    verifier=BoundaryVerifier(thresholds=Thresholds(uncertain_probability=.001,confirm_probability=.03))
    def measure(label):
     ts=time.perf_counter();qe=r.encode_panorama(query);pair=pair_from_encoding(r.runtime,qe,goalenc);scores=(torch.nn.functional.normalize(qe['global_descriptor'].float(),dim=-1)@gallery.T)[0];idx=gindex[t['trial_id']];order=torch.argsort(scores,descending=True);second=next((int(x) for x in order if int(x)!=idx),idx);margin=float(scores[idx]-scores[second]) if second != idx else float(scores[idx]);geom=lg.analyze(query,goal,pair['yaw_degrees']);second_pair=pair_from_encoding(r.runtime,qe,target_enc[second]) if second != idx else pair;sg=lg.analyze(query,targets[second][1],second_pair['yaw_degrees']) if second != idx else geom;lgmargin=geom['e_inliers']-sg['e_inliers'];top_candidate=targets[int(order[0])][0];target_top1=top_candidate==t['trial_id'];stable=target_top1 and (not history or history[-1]['top_candidate']==top_candidate);consistency=1.0 if stable else 0.0;e,vals=evidence(pair,geom,margin,lgmargin,stable,consistency);audit_position,audit_heading=state(env);audit_distance=float(np.linalg.norm((audit_position-target)[[0,2]]));row={'view_label':label,'timestamp_s':time.perf_counter()-started,'evidence':vals,'geometry':geom,'second_candidate_id':targets[second][0],'top_candidate':top_candidate,'target_top1':target_top1,'r361':pair,'position_gt_audit_only':audit_position.tolist(),'heading_gt_audit_only':audit_heading,'horizontal_distance_m_gt_audit_only':audit_distance,'formal_arrival_label_gt_audit_only':audit_distance<=1.0,'runtime_gt_inputs':[],'latency_ms':(time.perf_counter()-ts)*1000};history.append(row);views.append(e);lat.append(row['latency_ms']);return e,row
    e,row=measure('ORIGINAL');verifier.begin(e,0.0);verifier.state=Decision.UNCERTAIN_NEAR_BOUNDARY
    for phase,step_count,total_angle in [('LEFT_10',2,10.0),('RIGHT_20',4,-20.0),('RESTORE_10',2,10.0)]:
     angular=math.radians(total_angle)/(step_count*.1)
     for k in range(step_count):
      obs=env.step(vel(0,angular));ap,ah=state(env);steps.append({'request_id':t['request_id'],'candidate_node_id':t['target_node_id'],'collection_protocol':'ACTIVE_EVIDENCE_V2R2','action':'velocity_control','action_label':phase,'linear_mps':0.0,'angular_rps':angular,'control_source':'FIXED_MULTIVIEW_REOBSERVE','position_gt_audit_only':ap.tolist(),'heading_gt_audit_only':ah,'horizontal_distance_m_gt_audit_only':float(np.linalg.norm((ap-target)[[0,2]])),'runtime_gt_inputs':[],'set_agent_state_used':False,'bpl_updated':False,'csr_updated':False})
     query=np.asarray(obs['rgb'])[...,:3].copy();measure(phase)
    control_bearing=wrap(math.degrees(math.atan2(float(target[0]-p0[0]),float(-(target[2]-p0[2]))))-h0)
    while verifier.micro_actions < verifier.max_micro_actions:
     current_path=root/f'pre_{verifier.micro_actions}.png';Image.fromarray(query).save(current_path);trav=omni.ask({'command':'infer','image_path':str(current_path),'goal_heading_rad':math.radians(control_bearing),'goal_distance_m':2.0,'point_goal_active':True});safe=not bool(trav.get('error')) and float(trav.get('linear_velocity_mps',0))>0
     verifier.state=Decision.UNCERTAIN_NEAR_BOUNDARY
     if not verifier.can_approach(timestamp_s=time.perf_counter()-started,traversable=safe):break
     rot_steps=max(0,int(math.ceil(abs(math.radians(control_bearing))/.1)))
     for k in range(rot_steps):
      obs=env.step(vel(0,math.radians(control_bearing)/(max(rot_steps,1)*.1)));ap,ah=state(env);steps.append({'request_id':t['request_id'],'candidate_node_id':t['target_node_id'],'action':'velocity_control','action_label':'BOUNDARY_ROTATE','linear_mps':0.0,'angular_rps':math.radians(control_bearing)/(max(rot_steps,1)*.1),'control_source':'CONTROL_UPPER_BOUND_GT_BEARING','position_gt_audit_only':ap.tolist(),'heading_gt_audit_only':ah,'horizontal_distance_m_gt_audit_only':float(np.linalg.norm((ap-target)[[0,2]])),'runtime_gt_inputs':[],'set_agent_state_used':False,'bpl_updated':False,'csr_updated':False})
     for k in range(8):
      obs=env.step(vel(.2,0));ap,ah=state(env);steps.append({'request_id':t['request_id'],'candidate_node_id':t['target_node_id'],'action':'velocity_control','action_label':'BOUNDARY_APPROACH_015','linear_mps':.2,'angular_rps':0.0,'control_source':'OmniTrav traversability plus fixed nominal step','position_gt_audit_only':ap.tolist(),'heading_gt_audit_only':ah,'horizontal_distance_m_gt_audit_only':float(np.linalg.norm((ap-target)[[0,2]])),'runtime_gt_inputs':[],'set_agent_state_used':False,'bpl_updated':False,'csr_updated':False})
     query=np.asarray(obs['rgb'])[...,:3].copy();control_bearing=0.0;e,row=measure(f'APPROACH_{verifier.micro_actions+1}');verifier.after_approach(e,time.perf_counter()-started)
    pf,hf=state(env);dist=float(np.linalg.norm((pf-target)[[0,2]]));result={'trial_id':t['trial_id'],'request_id':t['request_id'],'candidate_node_id':t['target_node_id'],'collection_protocol':'ACTIVE_EVIDENCE_V2R2','verifier_decision_applied':False,'category_pre_registered':t['category_pre_registered'],'initial_position_gt_audit_only':p0.tolist(),'final_position_gt_audit_only':pf.tolist(),'final_horizontal_distance_m_gt_audit_only':dist,'formal_arrival_label_gt_audit_only':dist<=1.0,'final_decision':'EVIDENCE_ONLY_NO_CONFIRM','final_confirmed':False,'micro_approach_count':verifier.micro_actions,'nominal_approach_m':verifier.micro_actions*.15,'hold_seconds':time.perf_counter()-started,'views':history,'episode_initialization':'OFFLINE_CONTROLLED_COLLECTION_TERMINAL_STATE','episode_initialization_enters_evidence':False,'omnitrav_goal_distance_m':2.0,'omnitrav_goal_distance_source':'FIXED_CONTROL_PARAMETER','control_track':'CONTROL_UPPER_BOUND_GT_BEARING','control_bearing_enters_evidence':False,'runtime_verifier_gt_inputs':[],'set_agent_state_count':0,'bpl_update_count':0,'csr_update_count':0,**contract};results.append(result);print(json.dumps({'index':n+1,'total':len(subset),'trial':t['trial_id'],'collection':'EVIDENCE_ONLY','micro':verifier.micro_actions,'distance_audit':dist}),flush=True)
   finally:env.close()
 finally:omni.close()
 with open(run/'active_approach_results.jsonl','x') as f:
  for x in results:f.write(json.dumps(x,sort_keys=True)+'\n')
 with open(run/'active_approach_step_logs.jsonl','x') as f:
  for x in steps:f.write(json.dumps(x,sort_keys=True)+'\n')
 dump(run/'active_manifest.json',{'stage':'Integration V2 Stage I5','collection_protocol':'ACTIVE_EVIDENCE_V2R2','verifier_decision_applied':False,'absolute_episode_manifest_path':True,'source_project_written':False,'r361_device':'cpu','r361_precision':'fp32','geometry_device':f'cuda:{a.gpu}','run_id':a.run_id,'gpu':a.gpu,'trial_start':a.start,'trial_count':len(results),'effective_trial_manifest':str(SOURCE_OUT/'effective_development_trial_manifest.jsonl'),'effective_collection_registry':str(SOURCE_OUT/'effective_collection_registry.json'),'real_velocity_control':True,'active_approach_action_count':len(steps),'episode_initialization':'OFFLINE_CONTROLLED_COLLECTION_TERMINAL_STATE','episode_initialization_enters_evidence':False,'omnitrav_goal_distance_m':2.0,'omnitrav_goal_distance_source':'FIXED_CONTROL_PARAMETER','control_track':'CONTROL_UPPER_BOUND_GT_BEARING','control_bearing_enters_evidence':False,'set_agent_state_count':0,'runtime_verifier_gt_inputs':[],'bpl_update_count':0,'csr_update_count':0,'lightglue_r361_latency_p90_ms':float(np.percentile(lat,90)) if lat else None,'omnitrav_ready':omni.ready})
if __name__=='__main__':main()

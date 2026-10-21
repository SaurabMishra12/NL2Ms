from pathlib import Path
import sys,json,collections,numpy as np,pandas as pd,itertools
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]/'NL2Ms';R=ROOT/'kaggle_outputs/experiment';O=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from nl2ms import geometry,critical,signals

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
geo=json.load(open(R/'derived/geometry/layer_geometry.json'));d=pd.read_parquet(R/'sample_summary.parquet').set_index('sample_id');g={x['sample_id']:x for x in map(json.loads,open(R/'raw/generations/generations.jsonl'))}
ans={};bank={};posrows=[];nf=[];destats=[];recompute=[]
for p in sorted((R/'raw/hidden_states').glob('*.npz')):
 with np.load(p) as z:
  for key in z.files:
   if not key.endswith('::hidden'):continue
   sid=key.split('::')[0];h=z[key];positions=z[sid+'::positions'];e=np.load(R/'derived/entropy'/f'{sid}.npz');ai=int(e['meta::answer_index'][0]);target=int(e['meta::positions'][ai]);oldidx=min(1,h.shape[1]-1);newidx=int(np.flatnonzero(positions==target)[0]);answer=h[:,newidx,:].astype(float)
   posrows.append(dict(sample_id=sid,dataset=d.loc[sid,'dataset'],saved_bank_position=int(positions[oldidx]),answer_position=target,mismatch=bool(positions[oldidx]!=target),answer_is_last_prompt=bool(target==g[sid]['prompt_length']-1)))
   bank[sid]=np.nan_to_num(h[:,oldidx,:].astype(np.float32),nan=0,posinf=65504,neginf=-65504)
   for layer in range(h.shape[0]):
    a=h[layer];bad=int((~np.isfinite(a)).sum())
    if bad:nf.append(dict(sample_id=sid,layer=layer,n_nonfinite=bad,size=a.size))
   pp={k.split('::',1)[1]:e[k] for k in e.files if k.startswith('profiles::')}
   if (R/'derived/j_space'/f'{sid}.npz').exists():
    with np.load(R/'derived/j_space'/f'{sid}.npz') as j:pp['jspace_amplification']=np.r_[np.nan,j['prof::amplification_mean']]
   det=critical.detect_all(pp);co=critical.consensus(det,29)
   recompute.append(dict(sample_id=sid,saved=int(d.loc[sid,'critical_layer_consensus']),recomputed=co.critical_layer_consensus,shape=co.dominant_shape))
   # Sensitivity to removing embedding/early-boundary profile rows. This is a diagnostic, not a new discovery threshold.
   dd={}
   for start in [0,3,5]:
    ds={}
    for name,(key,absolute) in critical.DETECTOR_SPECS.items():
     if key not in pp:continue
     a=np.array(pp[key],float);a[:start]=np.nan
     if absolute:a=np.abs(a)
     if np.isfinite(a).any():ds[name]=int(np.nanargmax(a))
    dd[str(start)]=ds
   destats.append(dict(sample_id=sid,peak_locations_by_excluded_prefix=dd))
   e.close()
sel=np.random.default_rng(0).choice(len(geo['sample_ids']),size=256,replace=False);ids=[geo['sample_ids'][i] for i in sel]
print('Recompute all 25 stored geometry profiles',flush=True)
H=np.stack([bank[s] for s in geo['sample_ids']]);labels=np.array([g[s]['correct'] for s in geo['sample_ids']],dtype=object)
cfg=json.load(open(R/'config/experiment_config.json'))['geometry'];gg=geometry.geometry_across_layers(H,labels,cfg=SimpleNamespace(**cfg))
checks=[]
for k,v in gg['profiles'].items():
 orig=np.array(geo['profiles'][k]);diff=abs(v-orig);f=diff[np.isfinite(diff)]
 checks.append(dict(metric=k,match=bool(np.allclose(v,orig,rtol=1e-5,atol=1e-5,equal_nan=True)),max_abs_difference=float(f.max()) if len(f) else None))
# Paired correction recovered from saved target layers even when clipped offset labels overwrote the critical role.
f=pd.read_parquet(R/'interventions/intervention_outcomes.parquet');f['repaired_role']=f.layer_role;f.loc[f.layer==f.critical_layer,'repaired_role']='critical'
pairs=[];rng=np.random.default_rng(20261021)
for key,sub in [(('all',None),f),*list(f.groupby(['perturbation_kind','epsilon']))]:
 t=sub.groupby(['sample_id','repaired_role']).jsd_output.mean().unstack()[['critical','random_control']].dropna();delta=(t.critical-t.random_control).to_numpy();n=len(delta)
 boots=rng.choice(delta,(2000,n)).mean(1);signs=np.array(list(itertools.product([-1,1],repeat=n))) if n<=15 else rng.choice([-1,1],(10000,n));null=(signs*delta).mean(1);p=(np.sum(abs(null)>=abs(delta.mean()))+1)/(len(null)+1)
 pairs.append(dict(kind=key[0],epsilon=key[1],n_pairs=n,critical_mean=float(t.critical.mean()),random_mean=float(t.random_control.mean()),difference=float(delta.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),p_signflip=float(p)))
res=dict(hidden_position_audit=posrows,hidden_nonfinite_by_layer=nf,geometry_sample_ids_actual=ids,geometry_actual_dataset_coverage=dict(collections.Counter(s.split('-')[0] for s in ids)),geometry_recomputed_profiles=gg['profiles'],geometry_recompute_checks=checks,detector_recomputation=recompute,boundary_sensitivity=destats,intervention_repaired_roles_pairs=pairs,role_repair_note='Select saved layer == saved critical_layer as critical before aggregation, preserving original labels in raw files. No new model calls. This repairs role bookkeeping only; broken open-vocabulary proxy and numerical validity remain unresolved.')
(O/'supplement_results.json').write_text(json.dumps(clean(res),indent=2,allow_nan=False));print('Supplement complete',len(posrows),sum(x['mismatch'] for x in posrows),flush=True)

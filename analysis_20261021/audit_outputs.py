#!/usr/bin/env python3
"""Offline, reproducible audit; never loads model weights or changes source evidence."""
from pathlib import Path
import base64, collections, csv, hashlib, io, json, math, sys, tarfile, warnings
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]/'NL2Ms'
SRC=ROOT/'kaggle_outputs'; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from nl2ms import signals, critical
from nl2ms.logit_lens import LENS_SCALARS
from nl2ms.attention import ATTENTION_STATISTICS
warnings.filterwarnings('ignore', category=RuntimeWarning)
RNG=np.random.default_rng(20261021)

def clean(x):
 if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
 if isinstance(x,np.generic): return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x): return None
 if x is pd.NA: return None
 return x

def dump(x,p): p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False))
def load(p): return json.loads(p.read_text())
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def desc(v):
 a=np.asarray(v,dtype=float).ravel(); f=a[np.isfinite(a)]
 return dict(n_total=len(a),n_finite=len(f),n_nan=int(np.isnan(a).sum()),n_posinf=int(np.isposinf(a).sum()),n_neginf=int(np.isneginf(a).sum()),mean=float(f.mean()) if len(f) else None,std=float(f.std(ddof=1)) if len(f)>1 else None,min=float(f.min()) if len(f) else None,q25=float(np.quantile(f,.25)) if len(f) else None,median=float(np.median(f)) if len(f) else None,q75=float(np.quantile(f,.75)) if len(f) else None,max=float(f.max()) if len(f) else None)
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054; mid=(k/n+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
 return [mid-half,mid+half]
def bh(items,key='p_value',out='q_bh'):
 valid=[(i,x[key]) for i,x in enumerate(items) if x.get(key) is not None and np.isfinite(x[key])];valid.sort(key=lambda t:t[1]); m=len(valid); q=1.
 for rank in range(m,0,-1):
  i,p=valid[rank-1];q=min(q,p*m/rank);items[i][out]=q;items[i]['family_size']=m

def compare(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
 o={'n_correct':len(x),'n_incorrect':len(y),'mean_correct':float(x.mean()) if len(x) else None,'mean_incorrect':float(y.mean()) if len(y) else None}
 if min(len(x),len(y))<5:return dict(o,status='insufficient_groups_minimum_5')
 u,p=stats.mannwhitneyu(x,y,alternative='two-sided');n=len(x)+len(y);sd=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(n-2));dif=x.mean()-y.mean()
 boot=np.mean(RNG.choice(x,(2000,len(x))),axis=1)-np.mean(RNG.choice(y,(2000,len(y))),axis=1)
 return dict(o,status='ok',mean_difference=float(dif),hedges_g=float(dif/sd*(1-3/(4*n-9))) if sd else None,cliffs_delta=float(2*u/(len(x)*len(y))-1),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_value=float(p))

csvfile=(OUT/'combined_results.csv').open('w',newline='')
cw=csv.DictWriter(csvfile,fieldnames=['cohort','record_type','source','sample_id','dataset','record_id','layer','position','metric','statistic','value','status']);cw.writeheader();csvcount=0

def emit(cohort,typ,source,sid='',dataset='',rid='',layer='',position='',metric='',statistic='value',value=None,status=''):
 global csvcount
 v=clean(value)
 if isinstance(v,(dict,list)):v=json.dumps(v,ensure_ascii=False,separators=(',',':'))
 if value is not None and isinstance(value,(float,np.floating)) and not np.isfinite(value):status='NaN' if np.isnan(value) else ('+Infinity' if value>0 else '-Infinity')
 cw.writerow(dict(cohort=cohort,record_type=typ,source=source,sample_id=sid,dataset=dataset,record_id=rid,layer=layer,position=position,metric=metric,statistic=statistic,value=v,status=status));csvcount+=1

def flatten(x,prefix=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from flatten(v,prefix+'/'+str(k))
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x):yield from flatten(v,prefix+'/'+str(i))
 else:yield prefix,x

result={'scope':'All files under NL2Ms/kaggle_outputs except .git, __pycache__, .pytest_cache; archived runs preserved separately, never pooled','date':'2026-10-21','seed':20261021,'cohorts':{},'array_inventory':[],'profile_recalculation_checks':[],'archives':[],'source_checks':{},'limitations':['No model execution; full vocabulary logits were not saved, so full-vocabulary entropy and KL/JSD cannot be independently recomputed from top-20 probabilities.','Descriptive summaries of damaged generations are diagnostic, not evidence of successful reasoning.','All statistical comparisons are exploratory; bootstrap resamples 2000; paired sign flips 10000; no confirmatory hypothesis was preregistered.']}
profiles={}; tables={}; gens={}; metric_names=set(); jprofiles=[]
for run in ['experiment','experiment_pilot','experiment_resume_test']:
 r=SRC/run; g=rows(r/'raw/generations/generations.jsonl'); sam=rows(r/('data/samples.jsonl' if run=='experiment' else 'data/pilot_samples.jsonl'));sm={x['sample_id']:x for x in sam};gens[run]=g
 d=pd.DataFrame(g);d['dataset']=d.sample_id.map(lambda s:sm[s]['dataset']);d['zero_fraction']=d.generated_ids.map(lambda x:x.count(0)/len(x) if x else np.nan);d['all_token_zero']=d.generated_ids.map(lambda x:len(x)>0 and set(x)=={0});d['majority_token_zero']=d.zero_fraction>.5
 rr={'n_samples':len(sam),'n_generations':len(g),'duplicate_generation_ids':int(d.sample_id.duplicated().sum()),'model_verification':load(r/'config/model_verification.json'),'by_dataset':[],'generation_diagnostics':d[['sample_id','dataset','zero_fraction','all_token_zero','majority_token_zero']].to_dict('records')}
 for ds,sub in [('all',d),*list(d.groupby('dataset'))]:
  n=len(sub);defined=sum(sm[s].get('ground_truth') is not None for s in sub.sample_id);c=int((sub.correct==True).sum());w=int((sub.correct==False).sum());a=c+w
  rr['by_dataset'].append(dict(dataset=ds,n=n,n_ground_truth=defined,n_correct=c,n_incorrect=w,n_ungraded=n-a,n_no_ground_truth=n-defined,parse_coverage=a/defined if defined else None,conditional_accuracy=c/a if a else None,conditional_accuracy_wilson95=wilson(c,a),end_to_end_success=c/defined if defined else None,end_to_end_success_wilson95=wilson(c,defined),all_token_zero=int(sub.all_token_zero.sum()),majority_token_zero=int(sub.majority_token_zero.sum()),length_terminated=int((sub.finish_reason=='length').sum()),generated_tokens=int(sub.generation_length.sum()),mean_generation_length=float(sub.generation_length.mean())))
 for p in r.rglob('*.parquet'):
  t=pd.read_parquet(p);source=str(p.relative_to(SRC));tables[source]=t
  for idx,row in t.iterrows():
   v=row.to_dict();sid=v.get('sample_id','');ds=v.get('dataset',sm.get(sid,{}).get('dataset',''))
   for k,val in v.items():emit(run,'saved_table',source,sid,ds,idx,v.get('layer',''),metric=k,value=val)
 for p in r.rglob('*.json'):
  for key,val in flatten(load(p)):emit(run,'saved_json',str(p.relative_to(SRC)),metric=key,value=val)
 for i,v in enumerate(g):
  for key,val in v.items():emit(run,'raw_generation',str(r.relative_to(SRC))+'/raw/generations/generations.jsonl',v['sample_id'],sm[v['sample_id']]['dataset'],i,metric=key,value=val)
 profiles[run]={}; nonfinite=[]; checks=[]; rederived=[]
 for idx,p in enumerate(sorted(r.rglob('*.npz'))):
  with np.load(p,allow_pickle=False) as z:
   source=str(p.relative_to(SRC));sid=p.stem;ds=sm.get(sid,{}).get('dataset','');isprof=p.parent.name=='entropy';pp={}
   for key in z.files:
    a=z[key];entry=dict(cohort=run,source=source,key=key,shape=list(a.shape),dtype=str(a.dtype))
    if a.dtype.kind in 'biufc':
     st=desc(a);entry.update(st)
     for stat,val in st.items():emit(run,'array_summary',source,sid,ds,metric=key,statistic=stat,value=val)
     if st['n_nan']+st['n_posinf']+st['n_neginf']:nonfinite.append(entry)
    result['array_inventory'].append(entry)
    if isprof and key.startswith('profiles::'):
     name=key.split('::',1)[1];pp[name]=a;metric_names.add(name)
     for l,val in enumerate(a):emit(run,'sample_layer_profile',source,sid,ds,layer=l,metric=name,value=val)
    elif isprof and key=='perpos::lens_scalars':
     saved_positions=z['meta::positions']
     for l in range(a.shape[0]):
      for pos in range(a.shape[1]):
       for k,name in enumerate(LENS_SCALARS):emit(run,'sample_position_layer_lens',source,sid,ds,layer=l,position=int(saved_positions[pos]),metric=name,value=a[l,pos,k])
    elif isprof and key=='perpos::attention_summary':
     for l in range(a.shape[0]):
      for k,name in enumerate(ATTENTION_STATISTICS):
       vals=a[l,:,:,k]
       for stat,val in desc(vals).items():emit(run,'attention_head_position_summary',source,sid,ds,layer=l+1,metric=name,statistic=stat,value=val)
    elif p.parent.name=='j_space' and key.startswith('prof::'):
     for l,val in enumerate(a):emit(run,'jspace_layer',source,sid,ds,layer=l,metric=key,value=val)
   if isprof:
    profiles[run][sid]=pp
    for name,arr in [('entropy_delta',signals.derivative(pp['entropy'])),('entropy_curvature',signals.curvature_1d(pp['entropy']))]:
     if name in pp:checks.append(dict(sample_id=sid,metric=name,match=bool(np.allclose(arr,pp[name],rtol=1e-5,atol=1e-5,equal_nan=True))))
    if 'order_margin' in pp:
     checks.append(dict(sample_id=sid,metric='order_margin_delta',match=bool(np.allclose(signals.derivative(pp['order_margin']),pp['order_margin_delta'],atol=1e-5,equal_nan=True))))
    cp=z['perpos::candidate_probs'] if 'perpos::candidate_probs' in z.files else None
    ai=int(z['meta::answer_index'][0])
    if cp is not None and cp.shape[-1]>1 and sm[sid].get('answer_spec_type')=='closed_set':
     # Ground-truth candidate label maps to the probability axis in dataset order.
     candidates=sm[sid]['candidates'];ci=next((i for i,c in enumerate(candidates) if c['label']==sm[sid]['ground_truth']),None)
     if ci is not None:
      recal=signals.order_parameter_closed_set(cp[:,ai,:],ci)
      for k,v in recal.items():
       name='order_'+k
       if name in pp:checks.append(dict(sample_id=sid,metric=name,match=bool(np.allclose(v,pp[name],rtol=1e-4,atol=1e-5,equal_nan=True))))
   elif p.parent.name=='hidden_states':
    for key in z.files:
     if not key.endswith('::hidden'):continue
     hs=key.split('::')[0];ep=r/'derived/entropy'/f'{hs}.npz'
     if not ep.exists():continue
     with np.load(ep) as e:
      target=int(e['meta::positions'][int(e['meta::answer_index'][0])]);hp=z[hs+'::positions'];match=np.flatnonzero(hp==target)
      if len(match):
       rec=signals.trajectory_metrics(z[key][:,match[0],:].astype(float))
       for k,v in rec.items():
        name='profiles::traj_'+k
        if name in e:
         diff=np.abs(v-e[name]);f=diff[np.isfinite(diff)];checks.append(dict(sample_id=hs,metric='hidden_recompute_traj_'+k,match=bool(np.allclose(v,e[name],rtol=.01,atol=.002,equal_nan=True)),max_abs_error=float(f.max()) if len(f) else None,precision_note='saved float16 vs original float32 residual'))
  if idx%100==0:print(run,'arrays',idx,flush=True)
 rr['nonfinite_arrays']=nonfinite;rr['recalculation_checks']=checks;rr['n_profile_samples']=len(profiles[run]);rr['n_checks']=len(checks);rr['n_check_mismatches']=sum(not x['match'] for x in checks)
 result['cohorts'][run]=rr

print('Aggregate metrics',flush=True)
main=tables['experiment/sample_summary.parquet']; idf=tables['experiment/interventions/intervention_outcomes.parquet']; main=main.copy();main['all_token_zero']=main.sample_id.map({x['sample_id']:set(x['generated_ids'])=={0} for x in gens['experiment']})
result['main_shapes']=main.dominant_transition_shape.value_counts().to_dict();result['main_consensus_depth_counts']=main.critical_layer_consensus.value_counts().sort_index().to_dict()
result['main_detector_locations']={k:main[k].value_counts(dropna=False).sort_index().to_dict() for k in main if k.startswith('critical_layer_') and not k.endswith('normalised')}
result['main_shape_by_degeneracy']=main.groupby(['all_token_zero','dominant_transition_shape']).size().reset_index(name='n').to_dict('records')
result['main_scalar_descriptives']=[]
for ds,sub in [('all',main),*list(main.groupby('dataset'))]:
 for k in sub.select_dtypes(include=np.number):
  result['main_scalar_descriptives'].append(dict(dataset=ds,metric=k,**desc(sub[k])))
result['profile_statistics']=[];result['profile_peak_comparisons']=[]
for run,pmap in profiles.items():
 gmap={x['sample_id']:x for x in gens[run]};dsmap={x['sample_id']:x['dataset'] for x in rows(SRC/run/('data/samples.jsonl' if run=='experiment' else 'data/pilot_samples.jsonl'))}
 for ds in sorted(set(dsmap.values())):
  ids=[sid for sid in pmap if dsmap[sid]==ds]
  for key in sorted(metric_names):
   vals=[pmap[s][key] for s in ids if key in pmap[s]]
   if not vals:continue
   mat=np.stack(vals)
   for l in range(mat.shape[1]):
    rr=dict(cohort=run,dataset=ds,metric=key,layer=l,**desc(mat[:,l]));result['profile_statistics'].append(rr)
   if run=='experiment':
    xx=[np.nanmax(pmap[s][key]) for s in ids if key in pmap[s] and gmap[s]['correct'] is True];yy=[np.nanmax(pmap[s][key]) for s in ids if key in pmap[s] and gmap[s]['correct'] is False]
    result['profile_peak_comparisons'].append(dict(dataset=ds,metric=key,**compare(xx,yy)))
# BH is applied within each dataset across all finite peak tests; separate answer-spec scales.
for ds in sorted(main.dataset.unique()):
 subset=[x for x in result['profile_peak_comparisons'] if x['dataset']==ds];bh(subset)
result['scalar_comparisons']=[]
for ds,sub in [('all_exploratory_pooled',main),*list(main.groupby('dataset'))]:
 for k in ['critical_layer_consensus_normalised','transition_strength','detector_spread','max_jsd','max_curvature','max_velocity','final_entropy','max_margin']:
  result['scalar_comparisons'].append(dict(dataset=ds,metric=k,**compare(sub.loc[sub.correct==True,k],sub.loc[sub.correct==False,k])))
 subset=[x for x in result['scalar_comparisons'] if x['dataset']==ds];bh(subset)
# Match controls by sample, kind and epsilon, then use the SAMPLE as inference unit.
paired=[]
for (kind,eps),sub in [(('all',None),None),*list(idf.groupby(['perturbation_kind','epsilon']))]:
 t=idf if sub is None else sub
 means=t.groupby(['sample_id','layer_role']).jsd_output.mean().unstack();valid=means[['critical','random_control']].dropna();diff=(valid.critical-valid.random_control).to_numpy();n=len(diff)
 if n:
  boots=RNG.choice(diff,(2000,n)).mean(axis=1); flips=(RNG.choice([-1,1],(10000,n))*diff).mean(axis=1);pv=(1+np.sum(abs(flips)>=abs(diff.mean())))/10001
  paired.append(dict(kind=kind,epsilon=eps,n_pairs=n,critical_mean=float(valid.critical.mean()),random_mean=float(valid.random_control.mean()),mean_difference=float(diff.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),p_value=float(pv),sample_differences=dict(zip(valid.index,diff))))
bh(paired[1:]);result['paired_interventions']=paired
result['intervention_diagnostics']={'n_rows':len(idf),'n_samples':idf.sample_id.nunique(),'datasets':dict(collections.Counter(s.split('-')[0] for s in idf.sample_id.unique())),'n_samples_with_critical_role':idf.loc[idf.layer_role=='critical','sample_id'].nunique(),'n_samples_missing_critical_role':idf.sample_id.nunique()-idf.loc[idf.layer_role=='critical','sample_id'].nunique(),'baseline_predictions':idf.baseline_prediction.value_counts(dropna=False).to_dict(),'perturbed_predictions':idf.perturbed_prediction.value_counts(dropna=False).to_dict(),'margin_baseline':desc(idf.margin_baseline),'margin_delta':desc(idf.margin_delta),'answer_changed':idf.answer_changed.value_counts(dropna=False).to_dict(),'top1_changed':idf.top1_changed.value_counts(dropna=False).to_dict(),'n_nonfinite_jsd':int((~np.isfinite(idf.jsd_output)).sum())}
result['intervention_group_statistics']=[]
for keys,sub in idf.groupby(['perturbation_kind','epsilon','layer_role']):
 for k in ['jsd_output','margin_delta','entropy_delta','top1_changed','answer_changed','delta_norm','residual_norm']:
  result['intervention_group_statistics'].append(dict(kind=keys[0],epsilon=keys[1],layer_role=keys[2],metric=k,**desc(sub[k])))
geo=load(SRC/'experiment/derived/geometry/layer_geometry.json');result['geometry_dataset_coverage']=dict(collections.Counter(x.split('-')[0] for x in geo['sample_ids']))
result['jspace_dataset_coverage']=dict(collections.Counter(p.stem.split('-')[0] for p in (SRC/'experiment/derived/j_space').glob('*.npz')))
result['saved_statistical_results']={str(p.relative_to(SRC)):load(p) for p in SRC.glob('experiment*/statistics/*.json')}
# Historical archives: every member hashed; extract only metadata in memory, never overwrite current files.
print('Inspect historical archives',flush=True)
for p in sorted(SRC.glob('*.tar.gz')):
 ar={'path':str(p.relative_to(SRC)),'sha256':sha(p),'members':[],'generation_snapshots':[]}
 with tarfile.open(p,'r:gz') as tf:
  for m in tf:
   if not m.isfile():continue
   f=tf.extractfile(m);h=hashlib.sha256();capture=m.name.endswith(('generations.jsonl','experiment_summary.json','model_verification.json','experiment_config.json'));buf=[]
   for b in iter(lambda:f.read(4*1024*1024),b''):
    h.update(b)
    if capture:buf.append(b)
   ar['members'].append({'path':m.name,'size':m.size,'sha256':h.hexdigest()})
   if capture:
    raw=b''.join(buf).decode('utf-8');obj=[json.loads(l) for l in raw.splitlines() if l.strip()] if m.name.endswith('jsonl') else json.loads(raw)
    if m.name.endswith('generations.jsonl'):
     ar['generation_snapshots'].append({'member':m.name,'n':len(obj),'n_correct':sum(x.get('correct') is True for x in obj),'n_incorrect':sum(x.get('correct') is False for x in obj),'all_token_zero':sum(set(x.get('generated_ids',[]))=={0} for x in obj),'records':obj})
    else:ar.setdefault('metadata',{})[m.name]=obj
 result['archives'].append(ar);print(p.name,len(ar['members']),flush=True)
# All source .py files must match the downloaded code snapshot before using source to interpret it.
result['source_checks']['python_source_matches_downloaded_snapshot']={str(p.relative_to(ROOT)):p.read_bytes()==(SRC/'NL2Ms'/p.relative_to(ROOT)).read_bytes() for p in (ROOT/'nl2ms').glob('*.py')}
result['source_checks']['graph_generation']='2026-10-21T04:37:01Z';result['source_checks']['graph_scope']='Verify tier source audit; raw output subtree excluded from graph and fully enumerated on filesystem'
for section in ['main_shapes','main_consensus_depth_counts','main_detector_locations','main_shape_by_degeneracy','main_scalar_descriptives','profile_statistics','profile_peak_comparisons','scalar_comparisons','paired_interventions','intervention_diagnostics','intervention_group_statistics','geometry_dataset_coverage','jspace_dataset_coverage']:
 for k,v in flatten(result[section]):emit('experiment' if section!='profile_statistics' else 'see_record','recomputed_result',section,metric=k,value=v)
for run,rr in result['cohorts'].items():
 for k,v in flatten(rr['by_dataset']):emit(run,'recomputed_accounting','raw_generations',metric=k,value=v)
csvfile.close();result['combined_csv_rows']=csvcount;result['profile_metric_names']=sorted(metric_names)
dump(result,OUT/'audit_results.json');print('Wrote results',csvcount,flush=True)

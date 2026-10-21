from pathlib import Path
import json,csv,collections,numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parent/'NL2Ms/kaggle_outputs/experiment';a=json.load(open(O/'audit_results.json'));e=json.load(open(O/'extended_results.json'));p=O/'ANALYSIS.md';text=p.read_text()
def table(rows,keys):
 def f(x):return 'NA' if x is None else (f'{x:.5g}' if isinstance(x,float) else str(x))
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(f(r.get(k)) for k in keys)+' |' for r in rows])
mis=[v for v in a['cohorts']['experiment']['recalculation_checks'] if not v['match']];e['hidden_recompute_mismatch_sample_ids']=sorted(set(v['sample_id'] for v in mis));e['hidden_recompute_mismatch_count']=len(e['hidden_recompute_mismatch_sample_ids']);e['hidden_sanitization_source']='extraction.py lines 372-373 applies torch.nan_to_num before persistence'
text=text.replace('All saved hidden-state tensors were finite in this scan; NaN logit diagnostics therefore do not justify claiming that hidden states themselves were NaN. The source can replace nonfinite bank values, but this audit found none in saved hidden tensors.', '**Saved hidden-state finiteness is misleading here.** `extraction.py` applies `torch.nan_to_num` before writing tensors. Recomputed trajectories disagree in nonfinite masks for 36 main-run samples (432 metric checks), even though finite coordinates closely agree. For example, `gsm8k-da9e9737918f` has NaN in its original final-layer norm profile but an all-zero saved vector at that answer position. The saved tensors have lost the original nonfinite pattern. This is preprocessing loss, not repaired measurement validity. Pilot and resume checks show analogous mismatches for 1 and 2 samples respectively.')
text=text.replace('A zero empirical null count is not proof of zero false-positive probability.', 'The raw AR(1) calibration is **4 sharp curves out of 12,000 (0.0333%)**, rounded to 0.0% in the saved report; it is not actually zero. Moreover, a single-curve sharpness rate and a majority/dominant classification over correlated detectors are different estimands.')
text=text.replace('The stored dominant shape is a vote over detector classifications,', '**12 of the 16 sharp cases (75%) are all-token-0 generations.** The stored dominant shape is a vote over detector classifications,')
text=text.replace('`finish_analysis.py`, `package_evidence.py`, `verify_bundle.py`','`finish_analysis.py`, `finalize_details.py`, `package_evidence.py`, `verify_bundle.py`')
# Stratified susceptibility is identifiable from the saved margin profiles.
samples=[json.loads(l) for l in open(R/'data/samples.jsonl')];ss=[]
for ds in sorted(set(x['dataset'] for x in samples)):
 vals=[]
 for x in samples:
  if x['dataset']!=ds:continue
  with np.load(R/'derived/entropy'/f"{x['sample_id']}.npz") as z:
   if 'profiles::order_margin' in z:vals.append(z['profiles::order_margin'].astype(float))
 if not vals:continue
 mat=np.stack(vals)
 for l in range(mat.shape[1]):
  arr=mat[:,l];arr=arr[np.isfinite(arr)];ss.append(dict(dataset=ds,layer=l,n_finite=len(arr),mean=float(arr.mean()) if len(arr) else None,variance_ddof0=float(arr.var()) if len(arr) else None,variance_ddof1=float(arr.var(ddof=1)) if len(arr)>1 else None))
e['susceptibility_by_dataset']=ss
null=a['saved_statistical_results']['experiment/statistics/null_models.json'];conf=a['saved_statistical_results']['experiment/statistics/confounds.json'];sus=a['saved_statistical_results']['experiment/statistics/susceptibility.json']
notes='''## 17. Null models, confounds, and susceptibility\n\nThese are the original saved tests, retained with their original denominators. Layer-shuffle tests use 500 permutations; label-shuffle tests use 1,000, despite the general configuration of 2,000. A small layer-shuffle p-value indicates ordering structure relative to this null, not successful reasoning or criticality. Shuffling individual layer values destroys smoothness, so a smooth ordered null remains necessary.\n\n'''
notes+=table([dict(test=k,observed=v.get('observed',v.get('observed_sharpness')),null_p95=v.get('null_p95'),p_value=v.get('p_value'),n_perm=v.get('n_perm'),exceeds_null=v.get('exceeds_null')) for k,v in null.items() if k!='detector_false_positive'],['test','observed','null_p95','p_value','n_perm','exceeds_null'])
notes+='\n\nConfound correlations target consensus depth; adjusted/matched group effects instead target transition strength. Do not mix the two outcomes. The adjusted effect becomes more negative (raw −0.108 to −0.393), so the reported negative “attenuation” is not disappearance of the effect. The matched subset has only 21 cases per group and a nonsignificant rank test (p≈0.291); no causal identification follows. Answer-token frequency was not measured.\n\n'
notes+=table([dict(confound=k,**v) for k,v in conf['correlations'].items()],['confound','n','spearman_rho','p_value','status'])
notes+='\n\nNorm/no-norm entropy-profile correlation averages 0.5248 over 600 records. This is based on a sparse five-layer no-norm control, and the pathway already failed. It is a warning about readout sensitivity, not proof that normalization causes every observed peak.\n\nThe saved pooled susceptibility uses 579 samples, including open-vocabulary GSM8K and closed-set tasks. This contradicts the intended separation of order-parameter scales. The stored susceptibility detector is at layer 25 (distributed), while the raw variance maximum is at layer 28; the detector preprocessing is not raw argmax. Dataset-specific variance profiles have now been recomputed and added to JSON/CSV, with both ddof=0 and ddof=1 explicitly labeled. Their maximum-variance layers are:\n\n'
notes+=table([max([r for r in ss if r['dataset']==ds and r['variance_ddof0'] is not None],key=lambda r:r['variance_ddof0']) for ds in sorted(set(x['dataset'] for x in ss))],['dataset','layer','n_finite','mean','variance_ddof0','variance_ddof1'])
notes+='\n\nJ-space epsilon diagnostics recomputed from saved probe descriptors:\n\n'
j=pd.DataFrame(e['jspace_epsilon_summary']);jtab=[]
for eps,sub in j.groupby('epsilon'):
 jtab.append(dict(epsilon=float(eps),n_samples=len(sub),mean_amplification=float(sub['mean'].mean()),mean_peak_amplification=float(sub['max'].mean()),median_peak_layer=float(sub.peak_layer.median()),mean_rank_correlation_to_primary=float(sub.spearman_vs_primary.mean()),nonfinite_descriptor_cells=int(sub.n_nonfinite.sum())))
e['jspace_epsilon_aggregate']=jtab;notes+=table(jtab,['epsilon','n_samples','mean_amplification','mean_peak_amplification','median_peak_layer','mean_rank_correlation_to_primary','nonfinite_descriptor_cells'])
notes+='\n\nNumerical verification summary: 600/600 detector consensus values reproduce; all 25 geometry profiles reproduce under the original flawed position rule; all 14 finite original rank-test p-values reproduce. Entropy derivatives and closed-set order metrics reproduce from saved intermediate arrays. Hidden-trajectory nonfinite-mask mismatches are explicitly retained, not passed as harmless rounding. None of these checks validates missing model outputs or restores pre-sanitization hidden states.\n'
text+='\n\n'+notes
p.write_text(text)
(O/'extended_results.json').write_text(json.dumps(e,indent=2,allow_nan=False))
def flat(x,p=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from flat(v,p+'/'+str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from flat(v,p+'/'+str(i))
 else:yield p,x
with (O/'combined_results.csv').open('a',newline='') as f:
 w=csv.writer(f)
 for section in ['hidden_recompute_mismatch_sample_ids','hidden_recompute_mismatch_count','hidden_sanitization_source','susceptibility_by_dataset','jspace_epsilon_aggregate']:
  for k,v in flat(e[section],section):w.writerow(['experiment','recomputed_result','extended_results','','','','','',k,'value',v,''])
print('Detailed findings finalized')

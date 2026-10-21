"""Write one strict JSON with chunked lossless source bytes and readable results."""
from pathlib import Path
import base64,hashlib,json,gzip,shutil
O=Path(__file__).resolve().parent;R=O.parent/'NL2Ms';S=R/'kaggle_outputs'

def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,float):
  import math
  if not math.isfinite(v):return None
 return v

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(3*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[(str(p.relative_to(R)),p) for p in S.rglob('*') if p.is_file() and not any(x in {'.git','__pycache__','.pytest_cache'} for x in p.relative_to(S).parts)]
files += [(str(p.relative_to(R)),p) for p in (R/'nl2ms').glob('*.py')]
files += [(str(p.relative_to(R)),p) for p in [R/'README.md',R/'build_notebook.py',R/'configs/example_config.json',R/'configs/README.md',R/'requirements.txt']]
files += [('audit/'+p.name,p) for p in O.glob('*.py')]
files += [('audit/'+p.name,p) for p in O.glob('*.png')]
files += [('audit/'+p.name,p) for p in [O/'ANALYSIS.md',O/'audit_results.json',O/'supplement_results.json',O/'extended_results.json'] if p.exists()]
# Keep the CSV in the single-file package using lossless gzip to avoid duplicating gigabytes of repeated labels.
with (O/'combined_results.csv').open('rb') as src,(O/'combined_results.csv.gz').open('wb') as dst:
 with gzip.GzipFile(fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz,4*1024*1024)
files.append(('audit/combined_results.csv.gz',O/'combined_results.csv.gz'))
manifest=[];seen=set();out=O/'all_results_and_raw_outputs.json'
with out.open('w') as f:
 f.write('{"schema":"nl2ms-evidence-bundle-v1","records":[\n');first=True
 def emit(obj):
  global first
  if not first:f.write(',\n')
  f.write(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False));first=False
 emit({'type':'documentation','scope':'Every scientific file in kaggle_outputs including four historical tar.gz archives; Git databases and Python caches excluded. Archives are preserved byte-for-byte and separately inventoried, not counted as independent experiments.','encoding':'For each file, concatenate base64-decoded blob_chunk records sharing its sha256 in index order. Identical content is stored once. Native JSON nonfinite values are null in parsed views; original exact bytes retain NaN/Infinity. CSV contains full saved tables, profiles, per-position lens values, raw generation records, array summaries and recomputed statistics. Dense tensors remain in original NPZ files, embedded losslessly.','csv':'audit/combined_results.csv.gz decompresses to combined_results.csv','verification':'Every embedded file is SHA-256 checked against its source; output JSON validated one record at a time by verify_bundle.py.'})
 for name in ['audit_results.json','supplement_results.json','extended_results.json']:
  if (O/name).exists():emit({'type':'computed_results','name':name,'data':json.load(open(O/name))})
 for run in ['experiment','experiment_pilot','experiment_resume_test']:
  for name in [('data/samples.jsonl' if run=='experiment' else 'data/pilot_samples.jsonl'),'raw/generations/generations.jsonl']:
   emit({'type':'readable_raw_records','cohort':run,'path':name,'data':[json.loads(l) for l in open(S/run/name) if l.strip()]})
 for i,(name,p) in enumerate(sorted(files)):
  digest=sha(p);entry={'type':'file','path':name,'sha256':digest,'size':p.stat().st_size};manifest.append(entry);emit(entry)
  if digest not in seen:
   seen.add(digest)
   with p.open('rb') as src:
    j=0
    for b in iter(lambda:src.read(3*1024*1024),b''):
     emit({'type':'blob_chunk','sha256':digest,'index':j,'base64':base64.b64encode(b).decode('ascii')});j+=1
  if i%200==0:print('packaged',i,'/',len(files),flush=True)
 f.write('\n]}\n')
(O/'bundle_manifest.json').write_text(json.dumps({'files':manifest,'unique_blobs':len(seen),'bundle_size_bytes':out.stat().st_size},indent=2))
print('BUNDLE',len(files),out.stat().st_size,flush=True)

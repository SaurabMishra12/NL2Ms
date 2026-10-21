from pathlib import Path
import json,hashlib,base64,csv
O=Path(__file__).resolve().parent
p=O/'all_results_and_raw_outputs.json';expected={};actual={};counts={};last={};n=0
with p.open() as f:
 assert f.readline()=='{"schema":"nl2ms-evidence-bundle-v1","records":[\n'
 for line in f:
  if line.strip()==']}':break
  line=line.strip()
  if line==',':continue
  if line.endswith(','):line=line[:-1]
  if not line:continue
  obj=json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)));n+=1
  if obj['type']=='file':expected[obj['sha256']]=obj['size']
  elif obj['type']=='blob_chunk':
   h=obj['sha256'];assert obj['index']==last.get(h,-1)+1;last[h]=obj['index'];b=base64.b64decode(obj['base64'],validate=True);actual.setdefault(h,hashlib.sha256()).update(b);counts[h]=counts.get(h,0)+len(b)
 else:raise AssertionError('No closing JSON delimiter')
 assert not f.read().strip()
for h,size in expected.items():
 if size==0:assert h==hashlib.sha256(b'').hexdigest()
 else:assert actual[h].hexdigest()==h and counts[h]==size
manifest=json.load(open(O/'bundle_manifest.json'));source_checks=[]
for row in manifest['files']:
 name=row['path'];src=O/name[6:] if name.startswith('audit/') else O.parent/'NL2Ms'/name
 h=hashlib.sha256()
 with src.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 assert h.hexdigest()==row['sha256'];source_checks.append(name)
# Validate CSV syntax and rectangularity without loading all rows into memory.
with (O/'combined_results.csv').open(newline='') as f:
 rd=csv.reader(f);header=next(rd);count=0
 for row in rd:assert len(row)==len(header);count+=1
res={'strict_json_records':n,'files_verified':len(source_checks),'unique_content_hashes_verified':len(expected),'csv_rows':count,'csv_columns':len(header),'status':'PASS','note':'Byte integrity and serialization verified; scientific validity limitations remain in ANALYSIS.md.'}
(O/'verification.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))

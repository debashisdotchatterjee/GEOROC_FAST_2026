"""GEOROC FAST: revised empirical-Bayes province classifier.
No synthetic substitution for unavailable real data. Pilot results are exploratory.
"""
import os
for _var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var]='1'
import re, io, csv, json, hashlib, shutil, time, traceback, itertools, sys, platform, warnings
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.linalg import helmert
from scipy.special import logsumexp, multigammaln
from scipy.stats import invwishart, norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile, urllib.parse, tempfile

CLASSES=['OIB','CFB','VAB']; DOI={'OIB':'WFJZKY','CFB':'WSTPOX','VAB':'PVFZCE'}
MAJOR=['SIO2','TIO2','AL2O3','FE2O3T','MGO','CAO','NA2O','K2O']; TRACE=['RB','SR','Y','ZR','NB','BA']; PANEL=MAJOR+TRACE
METHODS=['Hierarchical EB','Independent EB','Logistic mean','Random forest mean']
COLORS=dict(zip(METHODS+['Oracle'],['#007C83','#D55E00','#5868A8','#9A6B24','#333333']))
STYLE=dict(zip(METHODS+['Oracle'],['-','--','-.',':',(0,(5,2))]))
REAL_N=[1,2,5,10,25,50]; SIM_N=REAL_N+[100]
BASE='https://data.goettingen-research-online.de'
@dataclass
class Config:
    root:str='/content/georoc_fast_results'
    cache:str='/content/georoc_fast_cache'
    source:str='mirror2021' # mirror2021, official13, manual13, or curated
    local_archive:str='' # directory or ZIP with version-13 raw files; names must contain DOI code
    curated_csv:str='' # previous audited analytic_candidates.csv.gz
    curation_file:str=''
    row_review_file:str=''
    reviewed:bool=False
    seed:int=20261126
    max_groups:int=20
    max_group_mb:float=12.
    max_rocks:int=250
    min_rocks:int=5
    min_provinces:int=10
    folds:int=5
    reps:int=10
    subsets:int=3
    rf_trees:int=50
    boot:int=500
    shrink:float=.25
    floor:float=.02
    oxide_min:float=85.
    oxide_max:float=105.
    era_cut:int=2000
    run_sim:bool=True
    run_real:bool=True
    auto_fallback:bool=True
    max_zip_mb:int=180
    download_seconds:int=55
    smoke:bool=False

def seed_for(c,*keys):
    return int.from_bytes(hashlib.sha256(str((c.seed,*keys)).encode()).digest()[:4],'little')
def save_table(df,root,name):
    p=Path(root)/'tables'; p.mkdir(parents=True,exist_ok=True)
    df.to_csv(p/f'{name}.csv',index=False)
    # HTML readable full tables; LaTeX-friendly input will use CSVs.
    if len(df)<=500:
        html=df.to_html(index=False,float_format=lambda x:f'{x:.5g}',escape=True)
        (p/f'{name}.html').write_text('<meta charset="utf-8"><style>body{font:14px Arial;margin:30px;color:#243344}table{border-collapse:collapse}td,th{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf3f6;position:sticky;top:0}</style><h2>'+name.replace('_',' ').title()+'</h2>'+html,encoding='utf8')
def atomic_csv(df,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(p)
def json_save(o,p): Path(p).write_text(json.dumps(o,indent=2,default=str),encoding='utf8')
def log(c,msg):
    print(msg,flush=True);p=Path(c.root);p.mkdir(parents=True,exist_ok=True)
    with (p/'run.log').open('a') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(msg)+'\n')
def figsave(fig,c,name):
    fig.tight_layout();p=Path(c.root)/'figures';p.mkdir(exist_ok=True,parents=True)
    for ext in ('png','pdf'):fig.savefig(p/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def digest_file(path,alg='sha256'):
    h=hashlib.new(alg)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def normtext(x):return re.sub(r'\s+',' ',str(x).strip()).upper()
def keytext(x):return re.sub(r'[^A-Z0-9]','',normtext(x))
def clean_header(x):return re.sub(r'\s+',' ',str(x).strip().lstrip('\ufeff')).upper()
def header_key(x):return re.sub(r'[^A-Z0-9%]','',clean_header(x))
def getcol(df,*names,required=False):
    keys={header_key(x):x for x in df.columns}
    for name in names:
        if header_key(name) in keys:return df[keys[header_key(name)]].fillna('').astype(str).str.strip()
    if required:raise ValueError(f'Missing {names}; columns: {list(df.columns)[:35]}')
    return pd.Series('',index=df.index,dtype=str)
def num(df,*names):
    # pd.to_numeric rejects <x, >x, BDL and malformed tokens; do not impute.
    x=pd.to_numeric(getcol(df,*names),errors='coerce');return x.where(np.isfinite(x))
def candidate_group(name,cls):
    x=re.sub(r'^.*?'+DOI[cls]+r'[ _-]+','',Path(name).stem,flags=re.I)
    x=re.sub(r'^\d{4}[-_]\d{2}[-_](?:\d{2}[-_])?','',x)
    x=re.sub(r'(?:[_ -]part[_ -]?\d+|[_ -]part[_ -]?[a-z])$','',x,flags=re.I)
    return cls+':'+x.upper()

def parse_raw(path):
    b=Path(path).read_bytes()
    try: txt=b.decode('utf-8-sig')
    except UnicodeDecodeError:txt=b.decode('cp1252',errors='strict')
    lines=txt.splitlines(keepends=True)
    header=next((i for i,x in enumerate(lines[:30]) if re.search(r'SAMPLE[ _]+NAME',x,re.I) and ('SIO2' in x.upper() or 'MATERIAL' in x.upper())),None)
    if header is None:raise ValueError(f'No recognizable header: {path}')
    stop=next((i for i,x in enumerate(lines[header+1:],header+1) if re.match(r'^\s*["\']?\s*(ABBREVIATIONS|REFERENCES)\s*:',x,re.I)),len(lines))
    delim='\t' if lines[header].count('\t')>lines[header].count(',') else ','
    # Fail loudly on malformed rows rather than on_bad_lines='skip'.
    with warnings.catch_warnings():
        warnings.simplefilter('error',pd.errors.ParserWarning)
        df=pd.read_csv(io.StringIO(''.join(lines[header:stop])),sep=delim,dtype=str,keep_default_na=False,on_bad_lines='error',index_col=False,low_memory=False)
    df.columns=[clean_header(x) for x in df.columns]
    # Extract local reference-number -> full bibliographic string, if possible.
    refs={}; in_ref=False
    for line in lines[stop:]:
        if re.match(r'^\s*["\']?\s*REFERENCES\s*:',line,re.I):in_ref=True;continue
        if not in_ref:continue
        text=line.strip().strip('"');match=re.match(r'^\s*\[?(\d+)\]?\s*[,;:\t ]+\s*(.*)',text)
        if match and match.group(2).strip():refs[match.group(1)]=match.group(2).strip('" ,')
    return df,refs,''.join(lines[stop:])

def reference_keys(citation,global_id,footer):
    # Global GEOROC reference IDs are preferred when explicitly present. Local [n] never treated as global.
    citation=str(citation);ids=re.findall(r'\[(\d+)\]',citation);out=set();details=[]
    if not ids and re.fullmatch(r'\s*\d+(?:\s*[,;]\s*\d+)*\s*',citation):ids=re.findall(r'\d+',citation)
    if global_id.strip() and re.fullmatch(r'\d+(?:\s*[,;|]\s*\d+)*',global_id):
        out.update('GID:'+x for x in re.findall(r'\d+',global_id))
    for i in ids:
        if i in footer: details.append(footer[i])
    # Short author-year strings are conservative cross-file links, not assumed unique publications.
    parts=re.split(r'(?=\[\d+\])',citation)
    for part in parts+details:
        part=re.sub(r'\[\d+\]','',part).strip(' ,;"')
        for doi in re.findall(r'10\.\d{4,9}/[^\s,;"<>]+',part,re.I):out.add('DOI:'+doi.rstrip(').').lower())
        m=re.search(r'([A-Za-zÀ-ÿ][^\d]{1,160}?)\s*[\(\[]?((?:19|20)\d{2})',part)
        if m:
            author=keytext(m.group(1));year=m.group(2)
            if author:out.add('AY:'+author+':'+year)
        if len(keytext(part))>20:out.add('TEXT:'+keytext(part))
    return '|'.join(sorted(out)), ' || '.join(details)

def curate(c,manifest):
    audit=Path(c.root)/'audit'; frames=[]; stats=[]; schemas=[]; mapping=[]
    for f in manifest.itertuples(index=False):
        raw,refs,footer=parse_raw(f.path); n=len(raw); group=candidate_group(f.file,f.cls)
        (audit/'references').mkdir(exist_ok=True);(audit/'references'/f'{f.cls}_{f.file}.txt').write_text(footer,encoding='utf8')
        schemas.append({'file':f.file,'columns':list(raw.columns),'parsed_rows':n,'footer_reference_ids':len(refs)})
        q=pd.DataFrame(index=raw.index)
        q['cls']=f.cls;q['file']=f.file;q['candidate_group']=group;q['province']=group;q['row_id']=f.cls+'/'+f.file+':'+(raw.index+1).astype(str)
        for name,aliases in {'sample':['SAMPLE NAME','SAMPLE'], 'location':['LOCATION'], 'location_comment':['LOCATION COMMENT'], 'rock_name':['ROCK NAME'], 'rock_type':['ROCK TYPE'], 'material':['MATERIAL','ANALYZED MATERIAL','ANALYSED MATERIAL','TYPE OF MATERIAL'], 'setting':['TECTONIC SETTING'], 'alteration':['ALTERATION','ALTERATION GRADE'], 'alteration_type':['ALTERATION TYPE'], 'citation':['CITATION','CITATIONS','REFERENCES','REFERENCE'], 'global_reference':['GEOROC REFERENCE ID','GEOROC CITATION ID','GLOBAL REFERENCE ID'], 'specimen_id':['IGSN','SAMPLE IGSN','GEOROC SAMPLE ID']}.items():
            q[name]=getcol(raw,*aliases,required=name in ['sample','rock_type','material','setting','citation'])
        for name,aliases in {'lat':['LATITUDE (MIN.)','LATITUDE MIN','LATITUDE'], 'lon':['LONGITUDE (MIN.)','LONGITUDE MIN','LONGITUDE'], 'lat_max':['LATITUDE (MAX.)','LATITUDE MAX'], 'lon_max':['LONGITUDE (MAX.)','LONGITUDE MAX'], 'loi':['LOI(WT%)','LOI (WT%)']}.items():q[name]=num(raw,*aliases)
        for col in [x for x in MAJOR if x!='FE2O3T']:q[col]=num(raw,col+'(WT%)')
        for col in TRACE:q[col]=num(raw,col+'(PPM)')
        fe3t=num(raw,'FE2O3T(WT%)','FE2O3TOTAL(WT%)','FE2O3(T)(WT%)');fet=num(raw,'FEOT(WT%)','FEOTOTAL(WT%)','FEO(T)(WT%)');fe3=num(raw,'FE2O3(WT%)');fe2=num(raw,'FEO(WT%)')
        factor=159.688/(2*71.844)
        # Both totals reported: exclude materially inconsistent duplicate totals (5% relative tolerance).
        conflict=fe3t.notna() & fet.notna() & ((fe3t-fet*factor).abs()>0.05*fe3t.abs())
        q['FE2O3T']=fe3t.combine_first(fet*factor).combine_first((fe3+fe2*factor).where(fe3.notna()&fe2.notna()&(fe3>=0)&(fe2>=0))).where(~conflict)
        q['iron_rule']=np.select([conflict,fe3t.notna(),fet.notna(),fe3.notna()&fe2.notna()],['conflicting totals','reported Fe2O3T','FeOT converted','separate FeO + Fe2O3'],default='unresolved')
        material=q.material.str.upper().str.replace(r'[^A-Z]','',regex=True).isin(['WHOLEROCK','WR','BULKROCK'])
        volcanic=q.rock_type.str.upper().str.strip().isin(['VOL','VOLCANIC','VOLCANIC ROCK','VOLCANIC ROCKS'])
        alt=q.alteration.str.upper().str.strip(); alt_type=q.alteration_type.str.upper().str.strip()
        fresh=alt.isin(['','F','FRESH','UNALTERED','NONE','NO','NOT ALTERED','UNKNOWN','NOT SPECIFIED','NA','N/A','ND']) & alt_type.isin(['','NONE','NO','NA','N/A','UNKNOWN','NOT SPECIFIED','UNALTERED'])
        positive=(q[MAJOR]>0).all(axis=1);q['oxide_sum']=q[MAJOR].sum(axis=1,min_count=len(MAJOR));q['normalized_silica']=100*q.SIO2/q.oxide_sum
        plausible=q.oxide_sum.between(c.oxide_min,c.oxide_max);basalt=q.normalized_silica.between(45,52)
        q['major_eligible']=material&volcanic&fresh&positive&plausible&basalt
        q['full_eligible']=q.major_eligible&(q[TRACE]>0).all(axis=1)&(q[TRACE]<=1e6).all(axis=1)
        q['chemical_exclusion']=np.select([~material,~volcanic,~fresh,~positive,~plausible,~basalt,~q.full_eligible],['not whole rock','not volcanic','documented or ambiguous alteration','major incomplete/nonpositive','oxide total outside bounds','outside normalized silica band','trace incomplete/nonpositive/out of units'],default='eligible')
        # Tectonic metadata agreement, NOT independent verification. Explicitly incompatible settings excluded by default.
        setting=q.setting.str.upper().str.strip().str.replace('_',' ',regex=False);pat={'OIB':r'OCEAN[ -]ISLANDS?|OCEANIC[ -]ISLANDS?','CFB':r'CONTINENTAL FLOOD BASALTS?','VAB':r'CONVERGENT MARGINS?|VOLCANIC ARCS?|ISLAND ARCS?|CONTINENTAL ARCS?|SUBDUCTION ZONES?'}[f.cls]
        q['metadata_agrees']=setting.str.fullmatch(pat,na=False)
        q['reference_keys']='';q['reference_details']=''
        # parse only chemically eligible candidates to save time, retaining raw audit counts
        for (cit,gid),idx in q.loc[q.major_eligible].groupby(['citation','global_reference'],sort=False).groups.items():
            keys,details=reference_keys(cit,gid,refs);q.loc[idx,'reference_keys']=keys;q.loc[idx,'reference_details']=details
        yr=q.citation.str.extract(r'(?<!\d)((?:19|20)\d{2})(?!\d)',expand=False);q['publication_year']=pd.to_numeric(yr,errors='coerce')
        mapping.append({'cls':f.cls,'file':f.file,'candidate_group':group,'province':group,'include':1,'verified':0,'evidence':'','notes':'Check metadata, original publications, split/related groups, and ambiguous records.'})
        stats.append({'cls':f.cls,'file':f.file,'raw_rows':n,'whole_rock':int(material.sum()),'volcanic_whole_rock':int((material&volcanic).sum()),'major_eligible':int(q.major_eligible.sum()),'full_eligible':int(q.full_eligible.sum()),'metadata_agrees_major':int((q.major_eligible&q.metadata_agrees).sum()),'missing_reference_major':int((q.major_eligible&(q.reference_keys=='')).sum())})
        # Metadata inventory helps geologist review without using chemistry to decide labels.
        frames.append(q.loc[q.major_eligible].copy())
    json_save(schemas,audit/'schemas.json');file_stats=pd.DataFrame(stats);save_table(file_stats,c.root,'data_file_audit');save_table(file_stats.groupby('cls',as_index=False).sum(numeric_only=True),c.root,'raw_class_audit')
    mapping=pd.DataFrame(mapping);atomic_csv(mapping,audit/'province_review_template.csv')
    data=pd.concat(frames,ignore_index=True)
    save_table(data.groupby(['cls','file','setting','alteration'],dropna=False).size().rename('n').reset_index(),c.root,'metadata_inventory')
    if c.curation_file:
        rev=pd.read_csv(c.curation_file,dtype=str,keep_default_na=False)
        need={'cls','file','province','include','verified','evidence'}
        if not need.issubset(rev):raise ValueError(f'Curation requires {need}')
        if rev.duplicated(['cls','file']).any():raise ValueError('Duplicate file mapping')
        if set(zip(rev.cls,rev.file))!=set(zip(mapping.cls,mapping.file)):raise ValueError('Review file must cover every manifest file exactly')
        if not rev.include.isin(['0','1']).all() or not rev.verified.isin(['0','1']).all():raise ValueError('include and verified must be 0 or 1')
        data=data.drop(columns='province').merge(rev[list(need)],on=['cls','file'],validate='many_to_one')
        if ((data.include=='1')&(data.province.str.strip()=='' )).any():raise ValueError('Empty province name')
        if c.reviewed and not ((rev.include=='0')|((rev.verified=='1')&(rev.evidence.str.strip()!=''))).all():raise ValueError('Reviewed mode requires evidence for every included file')
        data=data[data.include=='1'].copy();atomic_csv(rev,audit/'province_review_used.csv')
    elif c.reviewed:raise ValueError('reviewed=True requires a completed curation_file')
    data['metadata_include']=data.metadata_agrees
    data['manual_specimen']='';data['manual_refs']=''
    # Optional reviewed row decisions permit corrections only with evidence, no chemical-based relabeling.
    if c.row_review_file:
        rr=pd.read_csv(c.row_review_file,dtype=str,keep_default_na=False)
        required={'row_id','include','evidence','specimen_key','reference_keys'}
        if not required.issubset(rr) or rr.row_id.duplicated().any() or not rr.include.isin(['0','1']).all() or (rr.evidence.str.strip()=='').any():raise ValueError('Invalid row review file')
        if not set(rr.row_id).issubset(set(data.row_id)):raise ValueError('Unknown row review IDs')
        rr=rr.set_index('row_id');mask=data.row_id.isin(rr.index)
        data.loc[mask,'metadata_include']=data.loc[mask,'row_id'].map(rr.include).eq('1')
        data.loc[mask,'manual_specimen']=data.loc[mask,'row_id'].map(rr.specimen_key)
        data.loc[mask,'manual_refs']=data.loc[mask,'row_id'].map(rr.reference_keys)
        data.loc[data.manual_refs!='','reference_keys']=data.loc[data.manual_refs!='','manual_refs']
        atomic_csv(rr.reset_index(),audit/'row_review_used.csv')
    data.drop(columns=PANEL).to_csv(audit/'candidate_row_metadata.csv.gz',index=False,compression='gzip')
    pd.DataFrame(columns=['row_id','include','evidence','specimen_key','reference_keys']).to_csv(audit/'row_review_template.csv',index=False)
    if data.empty:raise ValueError('No eligible major-panel candidates; inspect audit')
    valid=data.metadata_include & (data.reference_keys!='') & (data['sample'].str.strip()!='') & (data.location.str.strip()!='')
    save_table(data.assign(valid=valid).groupby(['cls','province']).agg(chemical_candidates=('row_id','size'),metadata_and_reference_eligible=('valid','sum')).reset_index(),c.root,'candidate_province_counts')
    data=data.loc[valid].copy().reset_index(drop=True)
    if data.empty:raise ValueError('No rows with matching tectonic metadata and resolved references; review audit and row overrides')
    # Conservative duplicate reconciliation using specimen ID, shared citation+sample+location,
    # and exact chemical fingerprint with same sample/location. Keep all citation links.
    parent=list(range(len(data)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    seen={}
    for i,r in enumerate(data.itertuples(index=False)):
        sam=keytext(r.sample);loc=keytext(r.location);keys=[]
        if r.manual_specimen:keys.append(('MAN',r.manual_specimen))
        if r.specimen_id:keys.append(('ID',keytext(r.specimen_id)))
        keys.extend(('REFSAM',v,sam,loc) for v in r.reference_keys.split('|') if v)
        if r.full_eligible:keys.append(('CHEM',sam,loc,*[round(getattr(r,x),8) for x in PANEL]))
        for key in keys:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    data['duplicate_group']=[find(i) for i in range(len(data))];keep=[];dup_audit=[]
    counts_dup=data.duplicate_group.value_counts(); singleton=data.duplicate_group.map(counts_dup).eq(1)
    singles=data.loc[singleton].copy(); singles['replicate_count']=1
    for _,block in data.loc[~singleton].groupby('duplicate_group',sort=False):
        # Ambiguous grouping: exclude entire cluster; do not choose class via chemistry.
        if block.cls.nunique()!=1 or block.province.nunique()!=1:
            dup_audit.append({'rows':'|'.join(block.row_id),'action':'exclude cross-province/class cluster'});continue
        # Deterministic representative: prefer complete panel then archival row ID; no outcome-based selection.
        r=block.sort_values(['full_eligible','row_id'],ascending=[False,True]).iloc[0].copy()
        r['reference_keys']='|'.join(sorted(set(itertools.chain.from_iterable(x.split('|') for x in block.reference_keys))))
        r['replicate_count']=len(block);keep.append(r)
        if len(block)>1:dup_audit.append({'rows':'|'.join(block.row_id),'action':'retain '+r.row_id})
    save_table(pd.DataFrame(dup_audit,columns=['rows','action']),c.root,'duplicate_audit')
    data=pd.concat([singles,pd.DataFrame(keep)],ignore_index=True)
    if data.empty:raise ValueError('No unique specimens left')
    if (data.groupby('province').cls.nunique()>1).any():raise ValueError('Province contains conflicting class labels')
    data.to_csv(audit/'analytic_candidates.csv.gz',index=False,compression='gzip')
    save_table(data.groupby(['cls','province']).agg(unique_major=('row_id','size'),unique_full=('full_eligible','sum'),median_oxide_sum=('oxide_sum','median'),loi_available=('loi','count')).reset_index(),c.root,'unique_province_counts')
    save_table(data.assign(era=np.where(data.publication_year.isna(),'unknown',np.where(data.publication_year<c.era_cut,'before cutoff','at/after cutoff'))).groupby(['cls','province','era']).agg(major=('row_id','size'),full=('full_eligible','sum')).reset_index(),c.root,'publication_era_retention')
    save_table(data.groupby('cls').agg(unique_major=('row_id','size'),unique_full=('full_eligible','sum'),candidate_provinces=('province','nunique')).reset_index(),c.root,'unique_class_audit')
    data['y']=data.cls.map({x:i for i,x in enumerate(CLASSES)}).astype(int)
    return data

# ---- Mathematical model ----
def ilr(a):
    a=np.asarray(a,float)
    if not np.all(np.isfinite(a)&(a>0)):raise ValueError('ILR requires finite positive values')
    return np.log(a)@helmert(a.shape[1],full=False).T

def real_ilr(df,panel):
    a=df[panel].to_numpy(float).copy(); a[:,len(MAJOR):]/=1e4;return ilr(a)
def summary(x):
    x=np.asarray(x,float);m=x.mean(0);e=x-m;return len(x),m,e.T@e

def scale_fit(groups):
    c=np.mean([x.mean(0) for _,x in groups],axis=0);s=np.sqrt(np.mean([np.mean((x-c)**2) for _,x in groups]))
    if not np.isfinite(s) or s<=0:raise ValueError('Degenerate scale')
    return c,s


# ---------- Bounded acquisition: no repeated retries ----------
REPO='feog/DashVolcano'
MIRROR_DOI='10.1007/s00445-025-01882-5'
def get_json(url,params=None):
    r=requests.get(url,params=params,timeout=(5,12));r.raise_for_status();return r.json()
def download(url,path,c,params=None,limit_mb=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size: return path
    tmp=path.with_suffix(path.suffix+'.part');deadline=time.monotonic()+c.download_seconds
    try:
        with requests.get(url,params=params,stream=True,timeout=(5,12)) as r:
            r.raise_for_status();size=0
            with tmp.open('wb') as f:
                for b in r.iter_content(256*1024):
                    if time.monotonic()>deadline:raise TimeoutError('download time budget exceeded')
                    size+=len(b)
                    if size>(limit_mb or c.max_zip_mb)*1024**2:raise ValueError('download byte cap exceeded')
                    f.write(b)
        if size==0:raise ValueError('empty download')
        tmp.replace(path);return path
    finally:
        if tmp.exists():tmp.unlink()
def class_path(path):
    s=keytext(path)
    for k,patterns in {'OIB':['OCEANISLANDGROUP','OCEANISLANDCOMP','WFJZKY'],'CFB':['CONTINENTALFLOODBASALT','WSTPOX'],'VAB':['CONVERGENTMARGIN','PVFZCE']}.items():
        if any(p in s for p in patterns):return k
    return None
def select_groups(entries,c):
    # Select BEFORE looking at chemistry, labels beyond archive class, or model results.
    chosen=[];audit=[]
    for cls in CLASSES:
        a=[e for e in entries if e['cls']==cls]; groups={}
        for e in a:groups.setdefault(candidate_group(e['file'],cls),[]).append(e)
        eligible=[g for g,es in groups.items() if sum(e['size'] for e in es)<=c.max_group_mb*1024**2]
        rng=np.random.default_rng(seed_for(c,'archive selection',cls)); selected=set(rng.permutation(sorted(eligible))[:c.max_groups])
        for g,es in sorted(groups.items()):
            status='selected' if g in selected else ('size cap' if g not in eligible else 'random group cap')
            audit.append(dict(cls=cls,group=g,n_files=len(es),bytes=sum(e['size'] for e in es),status=status))
            if g in selected:chosen.extend(es)
    save_table(pd.DataFrame(audit),c.root,'archive_selection');return chosen

def mirror_entries(c):
    cache=Path(c.cache)/'mirror2021';cache.mkdir(parents=True,exist_ok=True)
    treefile=cache/'tree.json'
    if treefile.exists():tree=json.loads(treefile.read_text())
    else:
        # Resolve the branch to an immutable commit before downloading any data.
        commit=get_json(f'https://api.github.com/repos/{REPO}/commits/main')['sha']
        tree=get_json(f'https://api.github.com/repos/{REPO}/git/trees/{commit}',{'recursive':'1'});tree['commit']=commit
        if tree.get('truncated'):raise RuntimeError('GitHub tree is truncated')
        json_save(tree,treefile)
    commit=tree['commit'];items=[t for t in tree['tree'] if t['type']=='blob' and t['path'].startswith('GeorocDataset/') and class_path(t['path']) and t['path'].lower().endswith(('.zip','.csv'))]
    json_save({'repository':REPO,'commit':commit,'snapshot':'June 2021, as documented by repository','article_doi':MIRROR_DOI,'associated_data_doi':'10.21979/N9/BJENCK','license':'GEOROC source terms apply; consult archived source'},Path(c.root)/'audit'/'mirror_provenance.json')
    if set(class_path(t['path']) for t in items)!=set(CLASSES):raise RuntimeError('Could not locate all three class archives in GitHub tree; use official13 or manual13')
    # Only three relevant folder archives, not the full repository. Limit total size.
    zips=[t for t in items if t['path'].lower().endswith('.zip')];raw=[t for t in items if t['path'].lower().endswith('.csv')]
    if sum(t.get('size',0) for t in zips)>c.max_zip_mb*1024**2:raise RuntimeError('Mirror archive exceeds configured download budget; use official13 or manual13')
    def retrieve(t):
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/'+urllib.parse.quote(t['path'],safe='/');p=cache/(hashlib.sha256(t['path'].encode()).hexdigest()[:12]+'_'+Path(t['path']).name)
        p=download(url,p,c); h=hashlib.sha1(f'blob {p.stat().st_size}\0'.encode());
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        actual=h.hexdigest()
        if actual!=t['sha']:p.unlink();raise ValueError('Git blob hash mismatch: '+t['path'])
        return p
    entries=[]
    with ThreadPoolExecutor(max_workers=3) as ex:
        jobs={ex.submit(retrieve,t):t for t in zips}
        for fut in as_completed(jobs):
            t=jobs[fut];p=fut.result();cls=class_path(t['path']);log(c,f'Mirror archive ready: {t["path"]}')
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    if not i.is_dir() and i.filename.lower().endswith('.csv') and not Path(i.filename).name.startswith('._'):
                        entries.append(dict(cls=cls,file=Path(i.filename).name,size=i.file_size,zip=str(p),member=i.filename,origin=t['path'],version='2021-06 mirror '+commit))
    for t in raw:
        entries.append(dict(cls=class_path(t['path']),file=Path(t['path']).name,size=t['size'],git=t,origin=t['path'],version='2021-06 mirror '+commit))
    if len({(e['cls'],e['file']) for e in entries})!=len(entries):raise ValueError('Mirror contains duplicate archive member names; resolve before analysis')
    selected=select_groups(entries,c);out=[]
    for e in selected:
        if 'zip' in e:
            target=cache/e['cls']/e['file'];target.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(e['zip']) as z, z.open(e['member']) as f:
                with target.open('wb') as g:shutil.copyfileobj(f,g)
        else:target=retrieve(e['git'])
        out.append(dict(cls=e['cls'],file=e['file'],path=str(target),version=e['version'],origin=e['origin'],sha256=digest_file(target)))
    return out

def official_entries(c):
    cache=Path(c.cache)/'official13';cache.mkdir(parents=True,exist_ok=True);entries=[];fail=[]
    def meta(cls):
        code=DOI[cls];p=cache/f'{code}.json'
        if p.exists():m=json.loads(p.read_text())
        else:
            m=get_json(BASE+'/api/datasets/:persistentId/versions/13.0',{'persistentId':'doi:10.25625/'+code})['data'];json_save(m,p)
        if (m['versionNumber'],m['versionMinorNumber'])!=(13,0):raise ValueError('Wrong archive version')
        json_save(m,Path(c.root)/'audit'/f'{cls}_v13_metadata.json')
        result=[]
        for x in m['files']:
            d=x['dataFile'];name=Path(d.get('originalFileName') or x.get('label') or d['filename']).name
            if name.lower().endswith('.csv') and 'CITATION' not in name.upper():
                result.append(dict(cls=cls,file=name,size=int(d.get('filesize',d.get('fileSize',c.max_group_mb*1024**2+1)) or (c.max_group_mb*1024**2+1)),id=d['id'],original=bool(d.get('originalFileName') and d['originalFileName']!=d['filename']),checksum=d.get('checksum',{})))
        if not result:raise ValueError('No CSV metadata')
        return result
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs={ex.submit(meta,k):k for k in CLASSES}
        for f in as_completed(fs):
            try:entries.extend(f.result())
            except Exception as e:fail.append({'class':fs[f],'stage':'metadata','error':str(e)})
    if fail:save_table(pd.DataFrame(fail),c.root,'download_failures');raise RuntimeError('Official metadata unavailable; use manual13 or mirror2021')
    sel=select_groups(entries,c)
    def get(e):
        p=cache/e['cls']/e['file'];p=download(BASE+f'/api/access/datafile/{e["id"]}',p,c,{'format':'original'},c.max_group_mb+2)
        chk=e['checksum']; verified=False
        if chk.get('value') and not e['original']:
            verified=digest_file(p,chk['type'].lower().replace('-',''))==chk['value']
            if not verified:p.unlink();raise ValueError('checksum mismatch')
        return dict(cls=e['cls'],file=e['file'],path=str(p),version='13.0',origin=f'doi:10.25625/{DOI[e["cls"]]} file {e["id"]}',sha256=digest_file(p),repository_checksum_verified=verified)
    # One wave per class, modest concurrency; stop submitting after wall-clock budget.
    out=[];deadline=time.monotonic()+c.download_seconds
    for j in range(0,len(sel),8):
        if time.monotonic()>deadline:
            fail.extend(dict(cls=e['cls'],file=e['file'],stage='not attempted (time budget)',error='rerun or use manual data') for e in sel[j:]);break
        with ThreadPoolExecutor(max_workers=8) as ex:
            fs={ex.submit(get,e):e for e in sel[j:j+8]}
            for f in as_completed(fs):
                e=fs[f]
                try:out.append(f.result());log(c,f'Cached {e["file"]}')
                except Exception as err:fail.append(dict(cls=e['cls'],file=e['file'],stage='download',error=str(err)))
    save_table(pd.DataFrame(fail,columns=['cls','file','stage','error']),c.root,'download_failures')
    if fail:raise RuntimeError(f'{len(fail)} selected files unavailable; partial cache kept; no analysis of convenience subset. Rerun or use manual13.')
    return out

def manual_entries(c):
    p=Path(c.local_archive);entries=[]
    if not p.exists():raise FileNotFoundError('Set local_archive to downloaded v13 ZIP or folder')
    # Accept folder containing separate ZIPs or CSVs. Never extract arbitrary paths.
    paths=[p] if p.is_file() else list(p.rglob('*'))
    for f in paths:
        if f.suffix.lower()=='.zip':
            with zipfile.ZipFile(f) as z:
                for i in z.infolist():
                    cls=class_path(i.filename) or class_path(f.name)
                    if cls and i.filename.lower().endswith('.csv'):
                        entries.append(dict(cls=cls,file=Path(i.filename).name,size=i.file_size,zip=str(f),member=i.filename))
        elif f.suffix.lower()=='.csv' and class_path(str(f)):
            entries.append(dict(cls=class_path(str(f)),file=f.name,size=f.stat().st_size,path=str(f)))
    if set(e['cls'] for e in entries)!=set(CLASSES):raise ValueError('Need all three v13 archives; filenames should contain DOI codes or class directory names')
    out=[]
    for e in select_groups(entries,c):
        if 'zip' in e:
            target=Path(c.cache)/'manual13'/e['cls']/e['file'];target.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(e['zip']) as z,z.open(e['member']) as f,target.open('wb') as g:shutil.copyfileobj(f,g)
        else:target=Path(e['path'])
        out.append(dict(cls=e['cls'],file=e['file'],path=str(target),version='user-supplied v13 (not independently authenticated)',origin=str(e.get('zip',target)),sha256=digest_file(target)))
    return out

def acquire(c):
    modes=[c.source]+(['official13'] if c.source=='mirror2021' and c.auto_fallback else [])
    errors=[]
    for mode in modes:
        try:
            log(c,'Acquisition source: '+mode)
            rows={'mirror2021':mirror_entries,'official13':official_entries,'manual13':manual_entries}[mode](c)
            m=pd.DataFrame(rows)
            if m.empty or set(m.cls)!=set(CLASSES):raise RuntimeError('Not all classes available')
            if m.duplicated(['cls','file']).any():raise ValueError('Duplicate archive filenames')
            m.to_csv(Path(c.root)/'audit'/'archive_manifest.csv',index=False)
            if errors:save_table(pd.DataFrame(errors),c.root,'acquisition_errors')
            return m,mode
        except Exception as e:
            errors.append(dict(source=mode,error=str(e)));log(c,f'{mode}: {e}')
    save_table(pd.DataFrame(errors),c.root,'acquisition_errors');raise RuntimeError('No complete real dataset available. See acquisition_errors; no real data substituted.')

# ---------- Analytic empirical Bayes ----------
def sym(a):return (a+a.T)/2
def spd(a,floor):
    v,u=np.linalg.eigh(sym(a));return (u*np.maximum(v,floor))@u.T

def fit_eb(groups,c):
    """W: pooled unbiased within-province covariance; B_k: corrected means covariance,
    shrinkage and eigenvalue flooring. Gaussian prior for mu_k; exact conditional posterior.
    Covariance hyperparameters are plug-in estimates, not posterior draws."""
    sums=[(k,*summary(x)) for k,x in groups];d=len(sums[0][2]);I=np.eye(d)
    W0=sum(T for k,n,m,T in sums)/sum(n-1 for k,n,m,T in sums)
    wscale=max(np.trace(W0)/d,1e-10);W=spd((1-c.shrink)*W0+c.shrink*wscale*I,c.floor*wscale)
    raw=[]
    for k in range(3):
        a=[(n,m) for h,n,m,T in sums if h==k];N=np.array([n for n,m in a]);M=np.array([m for n,m in a]);
        if len(M)<2:raise ValueError('Need >=2 training provinces per class')
        # E[S_means]=B_k+W_true mean(1/n_g), also when n_g differ.
        raw.append(np.cov(M,rowvar=False,ddof=1)-W0*np.mean(1/N))
    bscale=max(np.mean([np.trace(b)/d for b in raw]),c.floor*wscale)
    fit=[]
    for k in range(3):
        B=spd((1-c.shrink)*raw[k]+c.shrink*bscale*I,c.floor*wscale)
        A=[(n,m) for h,n,m,T in sums if h==k];P=I/25.;r=np.zeros(d)
        for n,m in A:
            Q=np.linalg.solve(B+W/n,I);P+=Q;r+=Q@m
        V=np.linalg.solve(P,I);mu=V@r
        fit.append((mu,B,V))
    return W,fit

def lognormal(m,mu,A):
    d=len(m);L=np.linalg.cholesky(A);z=np.linalg.solve(L,m-mu);return -.5*(d*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+z@z)
def normlog(lp):return np.asarray(lp)-logsumexp(lp)
def eb_predict(fit,x):
    W,pars=fit;n=len(x);m=x.mean(0)
    # Residual scatter cancels because W is common and fixed after training.
    return normlog([lognormal(m,mu,B+V+W/n) for mu,B,V in pars])
def indep_predict(fit,x):
    # Matched ablation: same estimated class marginals and class means, no shared
    # province effect or shared mean uncertainty (V is included in each marginal). Product of individual Gaussian densities.
    W,pars=fit;n,m,T=summary(x);out=[];d=len(m)
    for mu,B,V in pars:
        C=B+W+V;D=m-mu;out.append(-.5*(n*(d*np.log(2*np.pi)+np.linalg.slogdet(C)[1])+np.trace(np.linalg.solve(C,T))+n*D@np.linalg.solve(C,D)))
    return normlog(out)
def fit_all_fast(groups,c,seed):
    fit=fit_eb(groups,c);x=np.concatenate([a for k,a in groups]);y=np.concatenate([np.full(len(a),k) for k,a in groups]);counts=np.bincount([k for k,a in groups],minlength=3)
    w=np.concatenate([np.full(len(a),1/(counts[k]*len(a))) for k,a in groups]);w*=len(w)/w.sum()
    lr=LogisticRegression(C=1.,max_iter=500).fit(x,y,sample_weight=w)
    rf=RandomForestClassifier(n_estimators=c.rf_trees,min_samples_leaf=5,max_features='sqrt',random_state=seed,n_jobs=1).fit(x,y,sample_weight=w)
    return fit,lr,rf

def ml_predict(m,x):return normlog(np.log(np.maximum(m.predict_proba(x).mean(0),1e-15)))
def scores(lq,y,method,**meta):
    p=np.exp(lq);return dict(**meta,method=method,y=int(y),pred=int(np.argmax(p)),nll=float(-lq[y]),brier=float(np.sum((p-np.eye(3)[y])**2)),correct=int(np.argmax(p)==y),confidence=float(p.max()),**{f'p{k}':float(p[k]) for k in range(3)})
def evaluate(fitted,x,y,ml_probs=None,**meta):
    f,lr,rf=fitted
    if ml_probs is None: ml_probs=(lr.predict_proba(x),rf.predict_proba(x))
    logs=[eb_predict(f,x),indep_predict(f,x)]+[normlog(np.log(np.maximum(p.mean(0),1e-15))) for p in ml_probs]
    return [scores(q,y,m,**meta) for q,m in zip(logs,METHODS)]

def simulate(c):
    rng0=np.random.default_rng(c.seed);rows=[];d=13;I=np.eye(d)
    scenarios=[(delta,tau,'Gaussian') for delta in (1.,2.,4.) for tau in (.15,.75,1.5)]+[(2.,tau,'Student t5') for tau in (.15,.75,1.5)]
    if c.smoke:scenarios=scenarios[:1]
    ns=[1,2,5,10,25,50,100]
    for idx,(delta,tau,law) in enumerate(scenarios):
        log(c,f'Simulation {idx+1}/{len(scenarios)}: separation {delta}, tau {tau}, {law}')
        mu=np.zeros((3,d));mu[1,0]=delta;mu[2,:2]=[delta/2,delta*np.sqrt(3)/2];B=[tau**2*I for _ in range(3)]
        for rep in range(c.reps):
            rng=np.random.default_rng(seed_for(c,'sim',idx,rep));train=[];test=[]
            def make(k,n):
                a=rng.multivariate_normal(mu[k],B[k]);e=rng.normal(size=(n,d));
                if law!='Gaussian':e*=np.sqrt(3/rng.chisquare(5,size=(n,1)))
                return a+e
            for k in range(3):
                train.extend((k,make(k,int(rng.integers(5,51)))) for _ in range(15 if not c.smoke else 3))
                test.extend((k,make(k,100)) for _ in range(6 if not c.smoke else 2))
            # Fit in the original standardized simulation coordinates; known oracle uses true parameters.
            fitted=fit_all_fast(train,c,seed_for(c,'fit',idx,rep))
            for j,(k,x) in enumerate(test):
                mp=(fitted[1].predict_proba(x),fitted[2].predict_proba(x))
                for n in ns:
                    a=x[:n];meta=dict(scenario=idx,delta=delta,tau=tau,law=law,rep=rep,province=j,n=n)
                    rows.extend(evaluate(fitted,a,k,ml_probs=[p[:n] for p in mp],**meta))
                    # Under t residuals this is explicitly a Gaussian moment benchmark, not a Bayes oracle.
                    q=normlog([lognormal(a.mean(0),mu[h],B[h]+I/n) for h in range(3)])
                    rows.append(scores(q,k,'Oracle' if law=='Gaussian' else 'Gaussian moment benchmark',**meta))
    df=pd.DataFrame(rows);save_table(df,c.root,'simulation_predictions');simulation_report(c,df);return df

def balanced_metrics(df,keys):
    # Collapse repeats within independent unit first (caller supplies pre-aggregated rows where needed).
    per=df.groupby(keys+['y'],as_index=False)[['nll','brier','correct','confidence']].mean()
    return per.groupby(keys,as_index=False)[['nll','brier','correct','confidence']].mean().rename(columns={'correct':'balanced_accuracy'})
def simulation_report(c,df):
    keys=['scenario','delta','tau','law','n','method','rep'];by=balanced_metrics(df,keys);save_table(by,c.root,'simulation_replicate_metrics')
    key=keys[:-1];agg=by.groupby(key,as_index=False).agg(nll=('nll','mean'),nll_mcse=('nll',lambda x:x.std(ddof=1)/np.sqrt(len(x))),brier=('brier','mean'),accuracy=('balanced_accuracy','mean'),accuracy_mcse=('balanced_accuracy',lambda x:x.std(ddof=1)/np.sqrt(len(x))),replicates=('rep','size'));save_table(agg,c.root,'simulation_summary')
    for law,block in agg.groupby('law'):
        b=block[block.delta==2] if (block.delta==2).any() else block;taus=sorted(b.tau.unique());fig,axs=plt.subplots(1,len(taus),figsize=(4.4*len(taus),3.6),squeeze=False)
        for ax,tau in zip(axs[0],taus):
            for method,a in b[b.tau==tau].groupby('method'):
                a=a.sort_values('n');ax.plot(a.n,a.nll,marker='o',label=method,color=COLORS.get(method,'#555555'));ax.set_xscale('log');ax.set_title(f'Between-province SD = {tau:g}');ax.set_xlabel('Rocks per test province');ax.set_ylabel('Balanced log loss')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'Simulation: {law}, separation {b.delta.iloc[0]:g}');figsave(fig,c,'simulation_'+law.replace(' ','_'))

# ---------- Real validation (no rock-level split) ----------
def tokens(x):return set(itertools.chain.from_iterable(str(v).split('|') for v in x))-{''}
def real_analysis(c,data,panel=PANEL,tag='primary'):
    data=data.copy();data['y']=data.cls.map({k:i for i,k in enumerate(CLASSES)})
    if data.y.isna().any():raise ValueError('Unexpected class')
    if len(panel)==len(PANEL):data=data[data.full_eligible.astype(str).str.lower().isin(['true','1'])].copy()
    data=data[np.isfinite(data[panel]).all(axis=1)&(data[panel]>0).all(axis=1)].copy()
    counts=data.groupby(['province','cls']).size().rename('available_rocks').reset_index();save_table(counts,c.root,'province_counts_'+tag)
    data=data[data.province.isin(counts.loc[counts.available_rocks>=c.min_rocks,'province'])].copy()
    if (data.groupby('province').cls.nunique()>1).any():raise ValueError('Conflicting group labels')
    units=data[['province','cls','y']].drop_duplicates().sort_values(['cls','province']).reset_index(drop=True);nc=units.cls.value_counts().reindex(CLASSES,fill_value=0)
    save_table(pd.DataFrame({'cls':CLASSES,'eligible_provinces':nc.values,'target':c.min_provinces}),c.root,'feasibility_'+tag)
    if nc.min()<3:log(c,f'{tag}: <3 units per class; no fit {nc.to_dict()}');return {'status':'insufficient groups','n_provinces':nc.to_dict()}
    label='reviewed' if c.reviewed else 'provisional archive units'
    if nc.min()<c.min_provinces:label+='; below prespecified province minimum'
    original_refs_for_test={p:tokens(g.reference_keys) for p,g in data.groupby("province")}
    # Cap rocks uniformly within each province, independent of their chemistry.
    parts=[]
    for p,g in data.groupby('province',sort=True):
        parts.append(g.sample(n=min(c.max_rocks,len(g)),random_state=seed_for(c,'cap',p)).sort_values('row_id'))
    data=pd.concat(parts,ignore_index=True);data['y']=data.y.astype(int)
    folds=min(c.folds,int(nc.min()));cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=c.seed)
    rows=[];audit=[];diag=[];fold_members=[]
    for fold,(itr,ite) in enumerate(cv.split(units,units.y),1):
        trp=set(units.iloc[itr].province);tep=set(units.iloc[ite].province);assert not trp&tep
        te=data[data.province.isin(tep)];tr0=data[data.province.isin(trp)]
        # Exclude known publication overlap using all eligible uncapped test records.
        refs=set().union(*(original_refs_for_test[p] for p in tep))
        crossing=tr0.reference_keys.map(lambda v:bool(refs.intersection(str(v).split('|'))));tr=tr0[~crossing].copy();ntr=tr.groupby('province').size();tr=tr[tr.province.isin(ntr[ntr>=c.min_rocks].index)]
        nt=tr.groupby('cls').province.nunique().reindex(CLASSES,fill_value=0)
        info=dict(fold=fold,training_rocks=len(tr),test_rocks=len(te),removed_crossing_rows=int(crossing.sum()),**{'train_'+k:int(nt[k]) for k in CLASSES})
        fold_members.extend(dict(fold=fold,province=p,role='test' if p in tep else 'train') for p in sorted(trp|tep))
        if nt.min()<2:audit.append(dict(**info,status='insufficient training groups after publication exclusion'));continue
        log(c,f'{tag}: fold {fold}/{folds}, {len(tep)} test units, {len(tr)} training rocks')
        train0=[(int(g.y.iloc[0]),real_ilr(g,panel)) for p,g in tr.groupby('province',sort=True)];center,scale=scale_fit(train0);train=[(k,(x-center)/scale) for k,x in train0];fitted=fit_all_fast(train,c,seed_for(c,tag,fold));W,pars=fitted[0]
        for k,(mu,B,V) in enumerate(pars):diag.append(dict(fold=fold,cls=CLASSES[k],within_trace=np.trace(W),between_trace=np.trace(B),mean_uncertainty_trace=np.trace(V),within_min_eigen=np.linalg.eigvalsh(W).min(),between_min_eigen=np.linalg.eigvalsh(B).min(),condition=np.linalg.cond(B+W),shrink=c.shrink))
        for p,g in te.groupby('province',sort=True):
            x=(real_ilr(g,panel)-center)/scale;y=int(g.y.iloc[0]);meta=dict(fold=fold,province=p,label_status=label)
            mp=(fitted[1].predict_proba(x),fitted[2].predict_proba(x));rows.extend(evaluate(fitted,x,y,ml_probs=mp,**meta,kind='full',n=len(x),subset=-1));rng=np.random.default_rng(seed_for(c,tag,p))
            for rep in range(c.subsets):
                order=rng.permutation(len(x))
                for n in REAL_N:
                    if n<=len(x):rows.extend(evaluate(fitted,x[order[:n]],y,ml_probs=[p[order[:n]] for p in mp],**meta,kind='subset',n=n,subset=rep))
        audit.append(dict(**info,status='ok'))
    save_table(pd.DataFrame(audit),c.root,'fold_audit_'+tag);save_table(pd.DataFrame(fold_members),c.root,'fold_members_'+tag);save_table(pd.DataFrame(diag),c.root,'covariance_diagnostics_'+tag)
    if not rows:return {'status':'no successful folds','n_provinces':nc.to_dict()}
    df=pd.DataFrame(rows);save_table(df,c.root,'real_predictions_'+tag);real_report(c,df,tag)
    return {'status':'exploratory' if (not c.reviewed or nc.min()<c.min_provinces or any(a['status']!='ok' for a in audit)) else 'completed; geological and statistical review required','n_provinces':nc.to_dict(),'predicted_provinces':int(df.province.nunique()),'all_folds_completed':all(a['status']=='ok' for a in audit),'label_status':label}

def real_report(c,df,tag):
    full=df[df.kind=='full'].copy();summ=balanced_metrics(full,['method']);save_table(summ,c.root,'real_metrics_'+tag)
    # Paired stratified province bootstrap (descriptive for overlapping CV fits).
    wide=full.pivot(index=['province','y'],columns='method',values='nll');rng=np.random.default_rng(seed_for(c,tag,'boot'));contr=[]
    for method in METHODS[1:]:
        dif=wide[method]-wide[METHODS[0]];samples=[dif.xs(y,level='y').to_numpy() for y in sorted(dif.index.get_level_values('y').unique())];draw=np.mean([rng.choice(v,(c.boot,len(v)),replace=True).mean(1) for v in samples],axis=0);contr.append(dict(comparator=method,delta_nll_comparator_minus_eb=np.mean([v.mean() for v in samples]),low=np.quantile(draw,.025),high=np.quantile(draw,.975)))
    save_table(pd.DataFrame(contr),c.root,'paired_bootstrap_'+tag)
    sub=df[df.kind=='subset'].copy()
    # Common cohort across n: largest n represented by all three classes (prefer >=3 each).
    candidates=[]
    for n,b in sub.groupby('n'):
        nc=b[['province','y']].drop_duplicates().y.value_counts().reindex(range(3),fill_value=0)
        if nc.min()>=1:candidates.append(int(n))
    if candidates:
        largest=max(candidates);cohort=set(sub.loc[sub.n==largest,'province']);sub=sub[(sub.province.isin(cohort))&(sub.n<=largest)]
        per=sub.groupby(['province','y','method','n'],as_index=False)[['nll','brier','correct','confidence']].mean();curves=balanced_metrics(per,['method','n']);save_table(curves,c.root,'common_cohort_curves_'+tag);save_table(per[['province','y']].drop_duplicates(),c.root,'curve_cohort_'+tag)
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for m,a in curves.groupby('method'):
            for ax,col,ylabel in zip(axs,['nll','confidence'],['Class-balanced log loss','Class-balanced confidence']):
                ax.plot(a.n,a[col],marker='o',color=COLORS[m],label=m);ax.set_xscale('log');ax.set_xlabel('Rocks per held-out province');ax.set_ylabel(ylabel)
        axs[0].legend(fontsize=8);fig.suptitle(f'Held-out provinces: {tag}, common cohort N={len(cohort)}');figsave(fig,c,'evidence_curves_'+tag)
    fig,axs=plt.subplots(1,4,figsize=(13,3.4))
    for ax,m in zip(axs,METHODS):
        b=full[full.method==m];cm=pd.crosstab(b.y,b.pred).reindex(index=range(3),columns=range(3),fill_value=0).to_numpy();ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()));ax.set_xticks(range(3),CLASSES);ax.set_yticks(range(3),CLASSES);ax.set_title(m,fontsize=10);ax.set_xlabel('Predicted');
        for i in range(3):
            for j in range(3):ax.text(j,i,cm[i,j],ha='center',va='center',color='white' if cm[i,j]>.55*cm.max() else 'black')
    axs[0].set_ylabel('Archive label');fig.suptitle(f'Province-level confusion matrices | {tag}');figsave(fig,c,'confusion_'+tag)
    fig,ax=plt.subplots(figsize=(5,4));bins=np.linspace(0,1,6);cal=[]
    for m,b in full.groupby('method'):
        b=b.assign(bin=np.minimum(np.searchsorted(bins,b.confidence,side='right')-1,4));z=b.groupby('bin').agg(confidence=('confidence','mean'),accuracy=('correct','mean'),n=('correct','size')).reset_index();z['method']=m;cal.append(z);ax.scatter(z.confidence,z.accuracy,s=15+5*z.n,label=m,color=COLORS[m]);
    ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='Mean maximum predicted probability',ylabel='Observed fraction correct',xlim=(0,1),ylim=(0,1),title='Descriptive top-label calibration');ax.legend(fontsize=8);figsave(fig,c,'calibration_'+tag);save_table(pd.concat(cal),c.root,'calibration_'+tag)
    fig,ax=plt.subplots(figsize=(6,3));a=pd.DataFrame(contr);y=np.arange(len(a));ax.errorbar(a.delta_nll_comparator_minus_eb,y,xerr=[np.maximum(a.delta_nll_comparator_minus_eb-a.low,0),np.maximum(a.high-a.delta_nll_comparator_minus_eb,0)],fmt='o',color='#007c83');ax.axvline(0,color='gray',ls='--');ax.set_yticks(y,a.comparator);ax.set_xlabel('Comparator minus EB log loss');ax.set_title('Paired province bootstrap (descriptive 95% intervals)');figsave(fig,c,'paired_'+tag)


def self_test(c):
    from scipy.stats import multivariate_normal
    rng=np.random.default_rng(11);d=3;train=[(k,rng.normal(k,1,(12,d))) for k in range(3) for j in range(4)];f=fit_eb(train,c);W,pars=f;x=rng.normal(size=(5,d));n=len(x)
    # Full joint model after integrating class mean and province random effect.
    full=[multivariate_normal.logpdf(x.ravel(),mean=np.tile(mu,n),cov=np.kron(np.ones((n,n)),B+V)+np.kron(np.eye(n),W)) for mu,B,V in pars]
    err=float(np.max(np.abs(eb_predict(f,x)-normlog(full))));assert err<1e-9
    # Verify independent ablation against direct individual densities.
    independent=[sum(multivariate_normal.logpdf(x,mean=mu,cov=B+V+W)) for mu,B,V in pars];err2=float(np.max(np.abs(indep_predict(f,x)-normlog(independent))));assert err2<1e-9
    assert all(np.linalg.eigvalsh(B).min()>0 and np.linalg.eigvalsh(V).min()>0 for mu,B,V in pars)
    a=rng.uniform(.1,2,(5,8));assert np.allclose(ilr(a),ilr(10*a));assert np.isclose(np.exp(eb_predict(f,x)).sum(),1.)
    # Saturation: covariance and predictive probabilities converge at fixed repeated mean.
    m=x.mean(0);lim=normlog([lognormal(m,mu,B+V) for mu,B,V in pars]);near=normlog([lognormal(m,mu,B+V+W/1e12) for mu,B,V in pars]);assert np.max(np.abs(lim-near))<1e-5
    json_save({'joint_log_probability_max_error':err,'independent_log_probability_max_error':err2,'positive_definiteness':True,'ilr_scale_invariance':True,'normalization':True,'saturation_limit':True},Path(c.root)/'numerical_self_tests.json')


def run(c):
    start=time.monotonic();root=Path(c.root);root.mkdir(parents=True,exist_ok=False)
    for d in ['tables','figures','audit']:(root/d).mkdir()
    import scipy,sklearn
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'pdf.fonttype':42,'figure.dpi':120,'savefig.dpi':300})
    json_save(asdict(c),root/'config.json');json_save({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},root/'environment.json')
    if '__file__' in globals() and Path(__file__).exists():shutil.copyfile(__file__,root/'georoc_fast.py')
    status={'method':'regularized empirical Bayes; no MCMC','simulation':'not run','real':'not run','publication_ready':False}
    try:
        with threadpool_limits(limits=1):
            self_test(c)
            if c.run_real:
                try:
                    if c.source=='curated':
                        data=pd.read_csv(c.curated_csv,keep_default_na=False);need={'cls','province','row_id','reference_keys','full_eligible',*PANEL};assert need.issubset(data.columns);data[PANEL]=data[PANEL].apply(pd.to_numeric,errors='coerce');status['source']='user-supplied previous curated table';status['curated_sha256']=digest_file(c.curated_csv)
                        if c.reviewed:raise ValueError('curated import cannot assert independent review; use raw files with review crosswalk')
                    else:
                        manifest,source=acquire(c);status['source']=source;log(c,'Curating selected real files');data=curate(c,manifest)
                    if not c.reviewed:status['label_warning']='Archive units provisional; not independently confirmed geological provinces'
                    status['real_primary']=real_analysis(c,data,PANEL,'primary');status['real_major_only']=real_analysis(c,data,MAJOR,'major_only');status['real']='finished; inspect per-panel statuses'
                except Exception as e:
                    status['real']='failed: '+str(e);(root/'audit'/'real_error.txt').write_text(traceback.format_exc());log(c,status['real'])
            if c.run_sim:
                try:simulate(c);status['simulation']='completed (synthetic experiment only)'
                except Exception as e:status['simulation']='failed: '+str(e);(root/'audit'/'simulation_error.txt').write_text(traceback.format_exc());log(c,status['simulation'])
    except Exception as e:status['fatal']=str(e);(root/'audit'/'fatal_error.txt').write_text(traceback.format_exc())
    finally:
        status['elapsed_seconds']=round(time.monotonic()-start,2);json_save(status,root/'run_status.json');archive=shutil.make_archive(str(root)+'_outputs','zip',root);print(json.dumps(status,indent=2));print('Results archive:',archive,flush=True)
    return archive,status

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='georoc_fast_'+time.strftime('%Y%m%d_%H%M%S'));p.add_argument('--cache',default='georoc_fast_cache');p.add_argument('--source',default='mirror2021',choices=['mirror2021','official13','manual13','curated']);p.add_argument('--local-archive',default='');p.add_argument('--curated-csv',default='');p.add_argument('--reps',type=int,default=10);p.add_argument('--no-real',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(Config(root=a.root,cache=a.cache,source=a.source,local_archive=a.local_archive,curated_csv=a.curated_csv,reps=1 if a.smoke else a.reps,rf_trees=10 if a.smoke else 50,run_real=not a.no_real,smoke=a.smoke))

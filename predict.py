"""Frozen missingness-aware H/A classifier; D evidence multiplied by 1.5; X preserved."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='2'
import argparse,csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import baseline
NAMES=['home_d1','home_d2','home_d3','home_nearest_speed','home_mean_speed','away_d1','away_d2','away_d3','away_nearest_speed','away_mean_speed','ball_speed','ball_speed_1s','ball_speed_3s','distance_touchline','distance_goalline','ball_missing','players_missing']
KEEP=list(range(10))+list(range(13,17))
MODEL_SHA256='e8cee0ffb48d6b1aad28abc79aeb7b8edc8eca354314329e3c723a3f9c9498a1'
def rolling(x,w):
 return baseline.rolling_mean(np.nan_to_num(x,nan=0.),w)

def features(players,ball,groups):
 # This function cannot access reference annotations.
 d=np.linalg.norm(players-ball[:,None,:],axis=2);d[~np.isfinite(d)]=1000
 vel=np.linalg.norm(np.diff(players,axis=0,prepend=players[:1]),axis=2)*25
 parts=[]
 for side in [0,1]:
  ds=d[:,groups==side];vs=vel[:,groups==side];nearest=ds.argmin(1)
  valid=np.isfinite(vs);mean=np.nansum(vs,axis=1)/np.maximum(valid.sum(1),1)
  parts.extend([np.sort(ds,axis=1)[:,:3],vs[np.arange(len(ball)),nearest,None],mean[:,None]])
 speed=np.linalg.norm(np.diff(ball,axis=0,prepend=ball[:1]),axis=1)*25
 parts.extend([speed[:,None],rolling(speed,12)[:,None],rolling(speed,37)[:,None],np.minimum(ball[:,1],68-ball[:,1])[:,None],np.minimum(ball[:,0],105.3-ball[:,0])[:,None],(~np.isfinite(ball).all(1))[:,None],(~np.isfinite(players).all(2)).mean(1)[:,None]])
 x=np.concatenate(parts,axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan
 assert x.shape[1]==len(NAMES)
 return x

def enriched(players,ball,groups):
 x=features(players,ball,groups)
 observed=np.isfinite(ball).all(1);idx=np.arange(len(ball));last=np.maximum.accumulate(np.where(observed,idx,-1))
 held=ball[np.maximum(last,0)].copy();held[last<0]=np.nan
 hx=features(players,held,groups)
 age=np.where(last<0,60.,np.minimum((idx-last)/25.,60.))[:,None]
 # Original binary features + nearest3 distances per team to last seen location + age.
 return np.c_[x[:,KEEP],hx[:,[0,1,2,5,6,7]],age].astype(np.float32)

def swap_enriched(x):
 return x[:,[5,6,7,8,9,0,1,2,3,4,10,11,12,13,17,18,19,14,15,16,20]]

def probability(model,x):
 return (model.predict_proba(x)+model.predict_proba(swap_enriched(x))[:,::-1])/2

def binary_decode(p):
 costs=np.array([[0.,16.],[16.,0.]]);e=np.log(np.maximum(p,1e-6));prev=np.zeros(e.shape,dtype=np.int8);score=e[0].copy()
 for i in range(1,len(p)):
  c=score[:,None]-costs;best=c.argmax(0);prev[i]=best;score=e[i]+c[best,np.arange(2)]
 path=np.empty(len(p),dtype=np.int8);path[-1]=score.argmax()
 for i in range(len(p)-1,0,-1):path[i-1]=prev[i,path[i]]
 return path

def fixed_dead_decode(probabilities,geometry_labels):
    # No annotations are accepted. Blocks are predicted live, not oracle live.
    out=geometry_labels.copy();live=geometry_labels!='D'
    bounds=np.flatnonzero(np.diff(np.r_[False,live,False]))
    for start,end in zip(bounds[::2],bounds[1::2]):
        # Canonicalize by baseline start team so exact Viterbi ties are equivariant.
        # Plain argmax would favor H in both the original and team-swapped input.
        orientation=int(geometry_labels[start]=='A')
        ordered=probabilities[start:end,::-1] if orientation else probabilities[start:end]
        out[start:end]=np.array(['H','A'])[binary_decode(ordered)^orientation]
    assert np.array_equal(out=='D',geometry_labels=='D')
    return out


def infer(data_dir):
    checkpoint=Path(__file__).with_name('augmented_model.pkl')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ValueError('Unexpected checkpoint hash; refusing pickle load')
    with checkpoint.open('rb') as f:model=pickle.load(f)
    tactical=Path(data_dir)/'tactical'
    players,ball,groups,names=baseline.load_tracking(tactical)
    phase,restart,before,kickoffs=baseline.load_tags(tactical/'BAR-ATH_tactical_smart_tagging.xml',names)
    original=np.full(baseline.FRAME_COUNT,'X');challenger=original.copy()
    for start,end in baseline.GRADED_HALVES:
        e=baseline.half_evidence(players,ball,groups,phase,restart,before,kickoffs,start,end)
        original[start:end]=baseline.decode(e)
        x=enriched(players[start:end],ball[start:end],groups)
        p=probability(model,x)
        challenger[start:end]=fixed_dead_decode(p,original[start:end])
    assert np.array_equal(challenger=='D',original=='D')
    assert np.array_equal(challenger=='X',original=='X')
    return challenger,original

def write_csv(path,labels):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['frame','label']);writer.writerows(enumerate(labels))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-output',type=Path)
    args=parser.parse_args();start=time.perf_counter()
    with threadpool_limits(limits=2):labels,original=infer(args.data_dir)
    write_csv(args.output,labels)
    if args.baseline_output:write_csv(args.baseline_output,original)
    print(json.dumps({'rows':len(labels),'seconds':time.perf_counter()-start,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'D_mask_matches_modified_geometry':True,'changed_HA_frames':int((labels!=original).sum())}))
if __name__=='__main__':main()

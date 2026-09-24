import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
d="TRANSCRIPT_dataset_v2.0.0/"  # 先把 TRANSCRIPT_dataset_v2.0.0.zip (Zenodo 7982976) 解压到本目录
R=pd.read_csv(d+"ratings_mat.csv",index_col=0); I=pd.read_csv(d+"items.csv",index_col=0)[R.index]; U=pd.read_csv(d+"users.csv",index_col=0)[R.columns]
Y=R.values
def z(M):
    r=np.apply_along_axis(rankdata,0,M); s=r.std(0); s[s==0]=1; return (r-r.mean(0))/s
Ir,Ur=z(I.values),z(U.values); n=Ir.shape[0]
Sdrug=np.nan_to_num(Ir.T@Ir/n); Sdis=np.nan_to_num(Ur.T@Ur/n)
C=np.nan_to_num(Ir.T@Ur/n)
print("|spearman| AUC:", roc_auc_score((Y==1).ravel(), np.abs(C).ravel()))
# top-k extreme genes connectivity (up/down 100) cosine-like
def topk_cs(k=100):
    S=np.zeros_like(C)
    Iv,Uv=I.values,U.values
    for j in range(Uv.shape[1]):
        up=np.argsort(-Uv[:,j])[:k]; dn=np.argsort(Uv[:,j])[:k]
        S[:,j]=-(Ir[up].mean(0)-Ir[dn].mean(0))
    return S
print("top100 up/down connectivity AUC:", roc_auc_score((Y==1).ravel(), topk_cs().ravel()))
# 5-fold CV on known (nonzero) entries; unknown kept as negatives in eval (pos vs rest within test pos + all unknown)
rng=np.random.default_rng(42)
pos=np.argwhere(Y==1); folds=rng.integers(0,5,len(pos))
res={k:[] for k in ["popularity","drugKNN","disKNN","both","both+rev"]}
for f in range(5):
    Yt=(Y==1).astype(float); test=pos[folds==f]
    Yt[test[:,0],test[:,1]]=0
    mask=np.ones_like(Yt,bool); mask[Yt==1]=False  # eval on test pos vs all unknowns (exclude train pos)
    lab=np.zeros_like(Yt); lab[test[:,0],test[:,1]]=1
    def k(Sim,kk=10):
        S=Sim.copy(); np.fill_diagonal(S,-np.inf); idx=np.argsort(-S,1)[:,:kk]
        W=np.zeros_like(Sim); np.put_along_axis(W,idx,np.take_along_axis(Sim,idx,1).clip(0),1); return W
    pop=Yt.sum(0,keepdims=True)*0+Yt.sum(1,keepdims=True)
    dk=k(Sdrug)@Yt; sk=Yt@k(Sdis).T
    rr=lambda M: rankdata(M.ravel()).reshape(M.shape)
    sc={"popularity":pop,"drugKNN":dk,"disKNN":sk,"both":rr(dk)+rr(sk),"both+rev":rr(dk)+rr(sk)+rr(-C)}
    for kk,v in sc.items(): res[kk].append(roc_auc_score(lab[mask],v[mask]))
for kk,v in res.items(): print(f"CV {kk}: AUC {np.mean(v):.3f} ± {np.std(v):.3f}")

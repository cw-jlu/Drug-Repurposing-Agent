import pandas as pd, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
d="TRANSCRIPT_dataset_v2.0.0/"  # 先把 TRANSCRIPT_dataset_v2.0.0.zip (Zenodo 7982976) 解压到本目录
R=pd.read_csv(d+"ratings_mat.csv",index_col=0); I=pd.read_csv(d+"items.csv",index_col=0); U=pd.read_csv(d+"users.csv",index_col=0)
print(R.shape,I.shape,U.shape,(R.values==1).sum(),(R.values==-1).sum())
print("gene order same:", (I.index==U.index).all(), "NaN items/users:", I.isna().sum().sum(), U.isna().sum().sum())
print("zero-var drugs:", (I.std()==0).sum(), "zero-var diseases:", (U.std()==0).sum())
I=I[R.index]; U=U[R.columns]
Ir=np.apply_along_axis(rankdata,0,I.fillna(0).values); Ur=np.apply_along_axis(rankdata,0,U.fillna(0).values)
Ir=(Ir-Ir.mean(0))/Ir.std(0); Ur=(Ur-Ur.mean(0))/Ur.std(0)
S=np.nan_to_num(-(Ir.T@Ur)/Ir.shape[0])
Y=R.values
m=Y!=0
print("AUC pos vs neg(-1) only:", roc_auc_score(Y[m]==1,S[m]))
print("AUC pos vs all non-pos (unknown as neg):", roc_auc_score((Y==1).ravel(),S.ravel()))
print("AUC pos vs all using +spearman (sign flip):", roc_auc_score((Y==1).ravel(),-S.ravel()))
# per-disease AUC pos vs rest
aucs=[roc_auc_score(Y[:,j]==1,S[:,j]) for j in range(Y.shape[1]) if 0<(Y[:,j]==1).sum()<Y.shape[0]]
print("per-disease AUC (pos vs rest) n=%d mean=%.3f median=%.3f"%(len(aucs),np.mean(aucs),np.median(aucs)))
# popularity baseline: drug degree (leaky, just to see structure)
deg=(Y==1).sum(1,keepdims=True)*np.ones_like(S)
print("drug-popularity AUC (leaky upper ref):", roc_auc_score((Y==1).ravel(),deg.ravel()))
# how many diseases have >=1 neg; pos per disease distribution
print("pos per disease:", np.percentile((Y==1).sum(0),[0,25,50,75,100]))
print("diseases with >=1 pos:", ((Y==1).sum(0)>0).sum(), "drugs with >=1 pos:", ((Y==1).sum(1)>0).sum())
# NS-AUC-like: per disease with >=1 pos, pos vs unknown
from collections import Counter
# LUAD-ish check: diseases list sample
print(list(R.columns[:5]))
# random baseline check
rng=np.random.default_rng(0); print("random AUC:", roc_auc_score((Y==1).ravel(), rng.random(S.size)))
# Top-k hit: for diseases with pos, recall@10 of reversal
rec=[ (Y[np.argsort(-S[:,j])[:10],j]==1).sum()/(Y[:,j]==1).sum() for j in range(Y.shape[1]) if (Y[:,j]==1).sum()>0]
print("mean Recall@10 reversal:", np.mean(rec), " random expected:", 10/613)
# signature similarity (drug-drug) kNN collaborative: leaky-free check skipped

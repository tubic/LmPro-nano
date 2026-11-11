#!/usr/bin/env python3
"""ProFE-Net ddG：从已保存权重复现 5 折交叉验证指标（用于论文 Table 3 的可复现交付）。

不做任何训练：加载 outputs/ddg_<tag>_fold<f>/best_model.pt，在该折的官方验证集上重跑推理，
输出 per-fold / mean±SD / pooled 指标，并同时给出 all / single / multiple 三种口径与
raw / linear-corrected 两种 RMSE-MAE 口径。

划分严格复刻 rde/datasets/skempi_adapter.py::SkempiDataset：
  对过滤后的 SKEMPI v2 取 unique PDB_Code 排序，均分 K 段，第 i 段为第 i 折的验证结构。

用法:
  python reproduce_table3.py --tag FIX --gpu 0 --no_pret
  python reproduce_table3.py --tag F8  --gpu 0 --pret_prep
  # 多个 run 取集成:
  python reproduce_table3.py --tag FIX --ens_tags FIX S7 S13 --gpu 0 --no_pret
"""
import os, sys, argparse, time, json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train/rde_2025.yml")
ap.add_argument("--tag", required=True, help="outputs/ddg_<tag>_fold<f>/ 下的权重")
ap.add_argument("--ens_tags", nargs="*", default=None, help="若给出多个 tag，则对其预测取平均（集成）")
ap.add_argument("--gpu", default="0")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--pair_budget", type=float, default=2.0e6)
ap.add_argument("--patch_size", type=int, default=128)
ap.add_argument("--maxlen", type=int, default=900)
ap.add_argument("--no_pret", action="store_true", help="与训练时一致：x_pret 置零")
ap.add_argument("--pret_prep", action="store_true", help="与训练时一致：在 patch 上算骨干上下文")
ap.add_argument("--head_rich", type=int, default=None, help="须与训练时一致：1=多尺度头, 0=仅差值头")
ap.add_argument("--loss_alpha", type=float, default=None, help="须与训练时一致")
ap.add_argument("--backbone_chi_mode", choices=["none", "native", "mask_mut"], default=None,
                help="须与训练时一致")
ap.add_argument("--out", default=None)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, REPO); os.chdir(REPO)
import torch, yaml, numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

def d2n(d): return SimpleNamespace(**{k: d2n(v) for k, v in d.items()}) if isinstance(d, dict) else d
raw = yaml.safe_load(open(a.config))
if a.head_rich is not None or a.loss_alpha is not None:
    raw.setdefault("model", {}).setdefault("head", {})
    if a.head_rich is not None: raw["model"]["head"]["rich"] = bool(a.head_rich)
    if a.loss_alpha is not None: raw["model"]["head"]["loss_alpha"] = float(a.loss_alpha)
if a.backbone_chi_mode is not None:
    raw.setdefault("model", {})["backbone_chi_mode"] = a.backbone_chi_mode
cfg = d2n(raw); device = torch.device("cuda")
from rde.models.rde_2025 import model_factory
from rde.datasets.skempi_adapter import SkempiDataset, SKEMPIDatasetAdapter
from rde.utils.protein.constants import BBHeavyAtom
ds = raw["dataset"]; K = int(ds["num_cvfolds"])

def official_val_pdbs():
    df = pd.read_csv(ds["csv_path"], sep=';')
    df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed', 'Temperature'])
    df = df[(df['Affinity_mut_parsed'] > 0) & (df['Affinity_wt_parsed'] > 0)]
    df['PDB_Code'] = df['#Pdb'].apply(lambda x: x.split('_')[0])
    p = sorted(df['PDB_Code'].unique()); n = len(p); fs = n // K
    return [set(p[i*fs:((i+1)*fs if i < K-1 else n)]) for i in range(K)]
VAL = official_val_pdbs()

PAD = ["aa_wt","aa_mut","pos_heavyatom","mask_heavyatom","mask","chain_nb","res_nb",
       "mut_flag","phi","psi","chi","phi_mask","psi_mask","chi_mask"]

def patch_idx(s):
    pos, ma = s["pos_heavyatom"], s["mask_heavyatom"]; L = pos.shape[0]
    if a.patch_size <= 0 or L <= a.patch_size: return None
    ca = pos[:, BBHeavyAtom.CA, :]; cb = pos[:, BBHeavyAtom.CB, :].clone()
    nb = ~ma[:, BBHeavyAtom.CB].bool(); cb[nb] = ca[nb]
    sel = s["mut_flag"].bool()
    if int(sel.sum()) == 0: return None
    return torch.argsort(torch.cdist(cb, cb[sel]).min(dim=1)[0])[:a.patch_size].sort()[0]

t0 = time.time(); ALL = []
for split in ("train", "val"):
    base = SkempiDataset(csv_path=ds["csv_path"], pdb_dir=ds["pdb_dir"], cache_dir=ds["cache_dir"],
                         num_cvfolds=K, cvfold_index=3, split=split)
    dset = SKEMPIDatasetAdapter(base, processed_dir=ds["processed_dir"])
    for i in range(len(dset)):
        s = dset[i]
        if s is None: continue
        L = s["pos_heavyatom"].shape[0]
        if L > a.maxlen or s["aa_wt"].shape[0] != L or int(s["mut_flag"].sum()) == 0: continue
        p = patch_idx(s)
        if p is not None:
            for k in PAD:
                if k in s: s[k] = s[k][p].contiguous()
        s["_pdb"] = s["id"]; s["_nmut"] = len(str(s["metadata"]["mutstr"]).split(","))
        ALL.append(s)
print(f"[data] {len(ALL)} 条, 用时 {time.time()-t0:.0f}s", flush=True)

def batches(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["aa_wt"].shape[0])
    ch, cur = [], []
    for i in order:
        N = rows[i]["aa_wt"].shape[0]
        cap = max(1, min(a.bs, int(a.pair_budget // max(1, N*N))))
        cur.append(i)
        if len(cur) >= cap: ch.append(cur); cur = []
    if cur: ch.append(cur)
    return ch

def collate(rows, idxs, model):
    grp = [rows[i] for i in idxs]; L = max(g["aa_wt"].shape[0] for g in grp); out = {}
    for k in PAD:
        if k not in grp[0]: continue
        buf = grp[0][k].new_zeros((len(grp), L) + tuple(grp[0][k].shape[1:]))
        for i, g in enumerate(grp): buf[i, :g[k].shape[0]] = g[k]
        out[k] = buf.to(device)
    if a.no_pret:
        z = torch.zeros(len(grp), L, 256, device=device)
        out["x_pret_wt"] = z; out["x_pret_mt"] = z
    # pret_prep: 不提供 -> 模型内部在该 patch 上实时计算（与训练等价，骨干冻结且确定性）
    out["ddG"] = torch.tensor([float(g["ddG"]) for g in grp], dtype=torch.float32, device=device)
    return out

@torch.no_grad()
def predict(tag, f, rows):
    ckpt = f"outputs/ddg_{tag}_fold{f}/best_model.pt"
    if not os.path.exists(ckpt): return None
    model = model_factory(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    P, T, NM = [], [], []
    for idxs in batches(rows):
        b = collate(rows, idxs, model)
        _, o = model(b)
        P += o["ddg"].flatten().tolist(); T += b["ddG"].tolist()
        NM += [rows[i]["_nmut"] for i in idxs]
    del model; torch.cuda.empty_cache()
    return np.array(P), np.array(T), np.array(NM)

def met(T, P):
    if T.size < 3 or P.std() < 1e-9: return None
    reg = LinearRegression().fit(P.reshape(-1,1), T); pc = reg.predict(P.reshape(-1,1))
    return dict(n=int(T.size), pearson=float(pearsonr(T,P)[0]), spearman=float(spearmanr(T,P)[0]),
                rmse_corrected=float(np.sqrt(((T-pc)**2).mean())), mae_corrected=float(np.abs(T-pc).mean()),
                rmse_raw=float(np.sqrt(((T-P)**2).mean())), mae_raw=float(np.abs(T-P).mean()),
                auroc=float(roc_auc_score((T>0).astype(int), P)) if len(set((T>0).tolist()))>1 else float('nan'))

tags = a.ens_tags if a.ens_tags else [a.tag]
per_fold, allP, allT, allNM = [], [], [], []
for f in range(K):
    ev = [s for s in ALL if s["_pdb"] in VAL[f]]
    if not ev: continue
    Ps = []
    for t in tags:
        r = predict(t, f, ev)
        if r is None: print(f"  [warn] 缺少 {t} fold{f} 权重"); continue
        Ps.append(r[0]); T, NM = r[1], r[2]
    if not Ps: continue
    P = np.mean(Ps, axis=0)
    m = met(T, P); m["fold"] = f; per_fold.append(m)
    allP.append(P); allT.append(T); allNM.append(NM)
    print(f"  fold{f}: n={m['n']:5d} P={m['pearson']:.4f} S={m['spearman']:.4f} "
          f"RMSE={m['rmse_corrected']:.3f} MAE={m['mae_corrected']:.3f} AUROC={m['auroc']:.4f}", flush=True)

KEYS = ["pearson","spearman","rmse_corrected","mae_corrected","rmse_raw","mae_raw","auroc"]
print(f"\n=== 折间 mean ± SD (审稿意见 R2#3) ===")
for k in KEYS:
    v = [m[k] for m in per_fold if m[k] == m[k]]
    print(f"  {k:16s} {np.mean(v):.4f} ± {(np.std(v, ddof=1) if len(v)>1 else 0.0):.4f}")
Pc, Tc, NMc = np.concatenate(allP), np.concatenate(allT), np.concatenate(allNM)
print(f"\n=== Pooled (论文口径: 每个点被测试一次), 三种突变数口径 ===")
for mode, sel in [("all", np.ones_like(NMc, bool)), ("single", NMc == 1), ("multiple", NMc > 1)]:
    m = met(Tc[sel], Pc[sel])
    print(f"  {mode:9s} n={m['n']:5d} P={m['pearson']:.4f} S={m['spearman']:.4f} "
          f"RMSE_c={m['rmse_corrected']:.3f} MAE_c={m['mae_corrected']:.3f} "
          f"RMSE_raw={m['rmse_raw']:.3f} AUROC={m['auroc']:.4f}")
print("\n[注] 与 Table 3 基线比较请用 *_corrected —— 基线取自 RDE-PPI 协议(先线性回归校正再算误差)。")
print(f"[配置] tags={tags} no_pret={a.no_pret} pret_prep={a.pret_prep} patch={a.patch_size} bs={a.bs}")
if a.out:
    json.dump({"tags": tags, "per_fold": per_fold}, open(a.out, "w"), indent=1)
    print(f"写入 {a.out}")

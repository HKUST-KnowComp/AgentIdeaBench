"""Honest F1 figure: 'saturation' is a fitting artifact.

Scatter cutoff vs Static (open/closed colored). Overlay:
  - linear fit (R^2=0.386)
  - bounded saturation curve forced k>=0.4 (R^2=0.289 — WORSE than linear)
Annotate: open-weight late slope ~0; closed-weight accelerating.
Output: reports/figures/honest_saturation.png
"""
import json, sqlite3, statistics, sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from reports._make_cross_year_plot import KNOWLEDGE_CUTOFFS

W = {"originality":2.0,"feasibility":1.0,"clarity":0.5,"impact":1.5,"specificity":0.5}
WS = sum(W.values())
FIG = ROOT/"reports"/"figures"

def wt(s): return sum(W[d]*s.get(d,0) for d in W)/WS
def r2(y,yh): return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
def closed(m): return m.startswith(("openai/","anthropic/","google/gemini"))

conn=sqlite3.connect(str(cfg.RESULTS_DB))
rows=conn.execute("SELECT idea_model,paper_id,scores_json FROM results WHERE track='B' AND prompt_version='v1_paper_refs' AND critic_model!='' AND scores_json IS NOT NULL").fetchall()
conn.close()
bk=defaultdict(list)
for m,p,sj in rows:
    try: bk[(m,p)].append(wt(json.loads(sj)))
    except: pass
mp=defaultdict(list)
for (m,p),v in {k:statistics.mean(v) for k,v in bk.items()}.items():
    mp[m].append(v)
data=[(m,KNOWLEDGE_CUTOFFS[m],statistics.mean(vs)) for m,vs in mp.items() if m in KNOWLEDGE_CUTOFFS]
data.sort(key=lambda r:r[1])

x=np.array([d[1] for d in data]); y=np.array([d[2] for d in data])
is_c=np.array([closed(d[0]) for d in data])

# fits
lin=np.polyfit(x,y,1); r2lin=r2(y,np.polyval(lin,x))
def sat(t,a,b,k): return a-b*np.exp(-k*(t-x.min()))
pb,_=curve_fit(sat,x,y,p0=[6.5,2.5,1.0],bounds=([5.5,0.5,0.4],[9,5,5]),maxfev=20000)
r2sat=r2(y,sat(x,*pb))

# split-half slopes (open only)
xo,yo=x[~is_c],y[~is_c]
med=np.median(xo)
se=np.polyfit(xo[xo<med],yo[xo<med],1)[0]
sl=np.polyfit(xo[xo>=med],yo[xo>=med],1)[0]
# closed slopes
xc,yc=x[is_c],y[is_c]
medc=np.median(xc)
sce=np.polyfit(xc[xc<medc],yc[xc<medc],1)[0]
scl=np.polyfit(xc[xc>=medc],yc[xc>=medc],1)[0]

fig,ax=plt.subplots(figsize=(10,6.2))
ax.scatter(xo,yo,s=85,c="#6b7280",edgecolors="#1f2937",lw=1.2,label=f"Open-weight (n={(~is_c).sum()})",zorder=3)
ax.scatter(xc,yc,s=95,c="#f59e0b",edgecolors="#92400e",lw=1.2,marker="D",label=f"Closed-weight (n={is_c.sum()})",zorder=3)
xl=np.linspace(x.min()-0.1,x.max()+0.1,100)
ax.plot(xl,np.polyval(lin,xl),"-",color="#2563eb",lw=2.3,label=f"Linear fit  (R²={r2lin:.3f})",zorder=2)
ax.plot(xl,sat(xl,*pb),"--",color="#dc2626",lw=2.3,label=f"Forced saturation curve  (R²={r2sat:.3f}, WORSE)",zorder=2)
ax.axhline(pb[0],color="#dc2626",ls=":",lw=0.9,alpha=0.5)
ax.annotate(f"forced asymptote {pb[0]:.2f}\n(artifact: free fit → straight line)",
            (x.max()-0.05,pb[0]),xytext=(-4,8),textcoords="offset points",
            fontsize=8.5,color="#991b1b",ha="right")
# label gpt-5/5.5
for m,cx,cy in data:
    if m in ("openai/gpt-5","openai/gpt-5.5"):
        ax.annotate(m.split("/")[-1],(cx,cy),xytext=(5,4),textcoords="offset points",fontsize=8,color="#92400e",fontweight="bold")
ax.set_xlabel("Knowledge cutoff (decimal year)",fontsize=11)
ax.set_ylabel("Static mean weighted score (1–10)",fontsize=11)
ax.set_title("F1 re-examined: 'saturation' is a fitting artifact\n"
             f"Open-weight late slope ≈ {sl:+.2f}/yr (flat), but closed-weight late slope ≈ {scl:+.2f}/yr (still rising)",
             fontsize=11.5)
ax.legend(loc="lower right",fontsize=9,frameon=False)
ax.grid(True,axis="y",alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(2,9)
plt.tight_layout()
out=FIG/"honest_saturation.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); plt.close()
print(f"linear R2={r2lin:.3f}  bounded-sat R2={r2sat:.3f}")
print(f"open late slope={sl:+.3f}  closed late slope={scl:+.3f}")
print(f"Wrote {out}")

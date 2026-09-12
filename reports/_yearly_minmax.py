"""F1 via yearly min/max — NO curve fit.

Per knowledge-cutoff year, show the range (min..max) of Static scores, with the
TOP split by open vs closed weight. Saturation = the top stops rising.
Output: reports/figures/yearly_minmax.png
"""
import json, sqlite3, statistics, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.rcParams["font.family"] = "Arial Unicode MS"  # CJK 覆盖
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
from reports._make_cross_year_plot import KNOWLEDGE_CUTOFFS

W = {"originality":2.0,"feasibility":1.0,"clarity":0.5,"impact":1.5,"specificity":0.5}
WS = sum(W.values())
FIG = ROOT/"reports"/"figures"
def wt(s): return sum(W[d]*s.get(d,0) for d in W)/WS
def closed(m): return m.startswith(("openai/","anthropic/","google/gemini"))

conn=sqlite3.connect(str(cfg.RESULTS_DB))
rows=conn.execute("SELECT idea_model,paper_id,scores_json FROM results WHERE track='B' AND prompt_version='v1_paper_refs' AND critic_model!='' AND scores_json IS NOT NULL").fetchall()
conn.close()
bk=defaultdict(list)
for m,p,sj in rows:
    try: bk[(m,p)].append(wt(json.loads(sj)))
    except: pass
mp=defaultdict(list)
for (m,p),v in {k:statistics.mean(v) for k,v in bk.items()}.items(): mp[m].append(v)
data=[(m,KNOWLEDGE_CUTOFFS[m],statistics.mean(vs)) for m,vs in mp.items() if m in KNOWLEDGE_CUTOFFS]

byyear=defaultdict(list)
for m,c,s in data: byyear[int(c)].append((m,s,closed(m)))
years=sorted(byyear)

ymin=[min(s for _,s,_ in byyear[y]) for y in years]
ymax=[max(s for _,s,_ in byyear[y]) for y in years]
open_max=[max([s for _,s,c in byyear[y] if not c], default=np.nan) for y in years]
clo_max =[max([s for _,s,c in byyear[y] if c],     default=np.nan) for y in years]
ns=[len(byyear[y]) for y in years]

fig,ax=plt.subplots(figsize=(9.5,6))

# min..max range bar per year (light gray)
for x,lo,hi in zip(years,ymin,ymax):
    ax.plot([x,x],[lo,hi],color="#d1d5db",lw=8,solid_capstyle="round",zorder=1)

# all model points
for y in years:
    for m,s,c in byyear[y]:
        ax.scatter(y,s,s=28,c="#f59e0b" if c else "#6b7280",
                   edgecolors="white",lw=0.5,zorder=2)

# top lines: open vs closed
ax.plot(years,open_max,"-o",color="#374151",lw=2.2,ms=8,label="Open-weight 最高分",zorder=4)
ax.plot(years,clo_max,"-D",color="#d97706",lw=2.2,ms=8,label="Closed-weight 最高分",zorder=4)
# floor line: overall min
ax.plot(years,ymin,"-s",color="#2563eb",lw=1.8,ms=7,label="每年最低分",zorder=4,alpha=0.85)

# annotate thin years
for x,n,hi in zip(years,ns,ymax):
    tag=f"n={n}"+("(单点,仅供参考)" if n<3 else "")
    ax.annotate(tag,(x,hi),xytext=(0,9),textcoords="offset points",
                ha="center",fontsize=8.5,color="#6b7280")

ax.set_xlabel("Knowledge cutoff 年份",fontsize=11)
ax.set_ylabel("Static 加权得分 (1–10)",fontsize=11)
ax.set_title("按年看最高/最低分:开源顶部 2024 后基本停涨(5.71→6.09),闭源仍升(6.90→7.17)",
             fontsize=11)
ax.set_xticks(years)
ax.set_ylim(2,9)
ax.grid(True,axis="y",alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="lower right",fontsize=9.5,frameon=False)
plt.tight_layout()
out=FIG/"yearly_minmax.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); plt.close()
print("year  n   min   max  open_max clo_max")
for i,y in enumerate(years):
    print(f"{y} {ns[i]:>3} {ymin[i]:>5.2f} {ymax[i]:>5.2f} {open_max[i]:>7.2f} {clo_max[i]:>7.2f}")
print("Wrote",out)

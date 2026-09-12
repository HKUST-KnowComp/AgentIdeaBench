"""Spindle (纺锤) view — between-model spread narrows→widens→narrows.

Left  : Static per-year distribution (box = IQR spindle body, whisker = min..max).
Right : frontier convergence — top-3 spread by year (Static vs Active).
Honest note: spindle holds for IQR / top-k, NOT for raw min-max (floor kept low by
weak late small models; closed ceiling still rising).
Output: reports/figures/spindle.png
"""
import json, sqlite3, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.rcParams["font.family"] = "Arial Unicode MS"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg
import reports._make_cross_year_plot as mc
KNOWLEDGE_CUTOFFS = mc.KNOWLEDGE_CUTOFFS
# axis = "cutoff"(默认) 或 "release"
AXIS = sys.argv[1] if len(sys.argv) > 1 else "cutoff"
TIME = mc.RELEASE_DATES if AXIS == "release" else KNOWLEDGE_CUTOFFS
AXIS_LABEL = "Release 年份" if AXIS == "release" else "Knowledge cutoff 年份"
W = {"originality":2.0,"feasibility":1.0,"clarity":0.5,"impact":1.5,"specificity":0.5}; WS=sum(W.values())
FIG = ROOT/"reports"/"figures"
def wt(s): return sum(W[d]*s.get(d,0) for d in W)/WS
def load(track):
    conn=sqlite3.connect(str(cfg.RESULTS_DB))
    rows=conn.execute("SELECT idea_model,paper_id,scores_json FROM results WHERE track=? AND prompt_version='v1_paper_refs' AND critic_model!='' AND scores_json IS NOT NULL",(track,)).fetchall();conn.close()
    bk=defaultdict(list)
    for m,p,sj in rows:
        try:bk[(m,p)].append(wt(json.loads(sj)))
        except:pass
    mp=defaultdict(list)
    for (m,p),v in {k:st.mean(v) for k,v in bk.items()}.items(): mp[m].append(v)
    return {m:st.mean(vs) for m,vs in mp.items() if m in TIME}

stat=load('B'); act=load('C')
def byyear(sc):
    d=defaultdict(list)
    for m,s in sc.items(): d[int(TIME[m])].append(s)
    return d
bs=byyear(stat); ba=byyear(act)
YEARS=sorted(y for y in bs if len(bs[y])>=3)   # 只取 Static n>=3 的年份

fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={"width_ratios":[1.25,1]})

# ---- LEFT: Static spindle (IQR box + min-max whisker + points) ----
for i,y in enumerate(YEARS):
    v=np.array(sorted(bs[y])); x=i
    q1,med,q3=np.percentile(v,[25,50,75])
    # min-max whisker (light)
    axL.plot([x,x],[v.min(),v.max()],color="#cbd5e1",lw=1.6,zorder=1)
    axL.plot([x-0.04,x+0.04],[v.min(),v.min()],color="#cbd5e1",lw=1.6)
    axL.plot([x-0.04,x+0.04],[v.max(),v.max()],color="#cbd5e1",lw=1.6)
    # IQR box = spindle body
    axL.add_patch(plt.Rectangle((x-0.18,q1),0.36,q3-q1,facecolor="#93c5fd",edgecolor="#1d4ed8",lw=1.4,zorder=2,alpha=0.85))
    axL.plot([x-0.18,x+0.18],[med,med],color="#1e3a8a",lw=2.2,zorder=3)
    # points
    axL.scatter(np.full(len(v),x)+np.random.RandomState(y).uniform(-0.05,0.05,len(v)),v,
                s=22,c="#475569",edgecolors="white",lw=0.4,zorder=4)
    axL.annotate(f"IQR={q3-q1:.2f}\nn={len(v)}",(x,v.max()),xytext=(0,8),
                 textcoords="offset points",ha="center",fontsize=8.5,color="#475569")
axL.set_xticks(range(len(YEARS))); axL.set_xticklabels(YEARS)
axL.set_xlabel(AXIS_LABEL,fontsize=11)
axL.set_ylabel("Static 加权得分 (1–10)",fontsize=11)
axL.set_title("Static 分布:核心差距(IQR)窄→宽→收窄\n(蓝箱=IQR 纺锤主体;细线=min–max)",fontsize=10.5)
axL.set_ylim(3,8); axL.grid(True,axis="y",alpha=0.3)
axL.spines[["top","right"]].set_visible(False)

# ---- RIGHT: frontier convergence (top-3 spread) ----
def top3spread(d,y):
    v=sorted(d.get(y,[]))[-3:]
    return (max(v)-min(v)) if len(v)>1 else np.nan
xs=YEARS
s_stat=[top3spread(bs,y) for y in xs]
s_act =[top3spread(ba,y) for y in xs]
axR.plot(xs,s_stat,"-o",color="#6b7280",lw=2.2,ms=9,label="Static top-3 差距")
axR.plot(xs,s_act,"-D",color="#10a37f",lw=2.2,ms=9,label="Active top-3 差距")
for x,yv in zip(xs,s_act):
    if not np.isnan(yv): axR.annotate(f"{yv:.2f}",(x,yv),xytext=(0,-14),textcoords="offset points",ha="center",fontsize=9,color="#0a5f4a",fontweight="bold")
for x,yv in zip(xs,s_stat):
    if not np.isnan(yv): axR.annotate(f"{yv:.2f}",(x,yv),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9,color="#374151")
axR.set_xticks(xs)
axR.set_xlabel(AXIS_LABEL,fontsize=11)
axR.set_ylabel("前沿(top-3)模型间差距",fontsize=11)
axR.set_title("前沿模型在收敛:top-3 差距 窄→宽→窄\n最新一年塌缩 → 现有 bench 区分不开顶尖模型",fontsize=10.5)
axR.set_ylim(0,1.6); axR.grid(True,axis="y",alpha=0.3)
axR.spines[["top","right"]].set_visible(False)
axR.legend(loc="upper right",fontsize=9.5,frameon=False)

plt.tight_layout()
out=FIG/f"spindle_{AXIS}.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); plt.close()
print(f"AXIS={AXIS}  YEARS={YEARS}")
print("Static min by year:",[round(min(bs[y]),2) for y in YEARS])
print("Static IQR by year:",[round(np.percentile(sorted(bs[y]),75)-np.percentile(sorted(bs[y]),25),2) for y in YEARS])
print("Static top3 spread:",[round(x,2) for x in s_stat])
print("Active top3 spread:",[round(x,2) for x in s_act])
print("Wrote",out)

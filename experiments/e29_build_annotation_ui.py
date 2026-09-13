"""E29 — Build the blind human-annotation frontend from reports/e29_human_eval_pairs.json.

Emits two self-contained (offline) HTML files, one per 50-pair set:
  reports/e29_annotation_S1.html   (pairs 1-50,  annotators A1/A2/A3)
  reports/e29_annotation_S2.html   (pairs 51-100, annotators A4/A5/A6)

Each file is BLIND: it embeds only {pair_id, subfield, left_text/left_text_zh,
right_text/right_text_zh}. No model names, no critic scores, no critic preference.

UI (2026-07-23 redesign per user):
  - SIX rating axes per pair: Overall + 5 dimensions (Originality/Feasibility/
    Clarity/Impact/Specificity), each judged Left / Tie / Right independently.
    Overall is required and keyboard-driven; the 5 dimensions are optional finer
    signal.
  - Bilingual (English + 中文) idea text and UI labels (annotators are NLP
    researchers); the Chinese idea translation is shown when present in the master.
  - Prev/Next at the TOP (sticky) with Enter=next and optional auto-advance, so
    paging is fast.
  - A bottom COMPLETION GRID visualizing which items are done (click a cell to
    jump), plus live stats.
  - Autosave to localStorage; export JSON + CSV. CSV columns:
      annotator,set,pair_id,subfield,overall,originality,feasibility,clarity,
      impact,specificity,confidence,note,ts   (each axis in {L,R,T,''}).

READ-ONLY on the master JSON; writes HTML to reports/ only. /usr/bin/python3.
Usage:
  /usr/bin/python3 experiments/e29_build_annotation_ui.py                 # NLP set (default)
  /usr/bin/python3 experiments/e29_build_annotation_ui.py --master reports/e29_human_eval_pairs.json
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
# Default to the NLP master (this frontend was redesigned for the NLP human eval);
# pass --master reports/e29_human_eval_pairs.json for the older CS/AI set.
MASTER = OUT / "e29_human_eval_pairs_nlp.json"

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentIdeaBench — NLP Pairwise Eval (__SETNAME__)</title>
<style>
  :root{--bg:#0f1115;--card:#191c23;--fg:#e6e8ec;--mut:#9aa3af;--acc:#4f8cff;--good:#2ea043;--warn:#d29922;--line:#2a2f3a;--zh:#8fb3ff;}
  @media (prefers-color-scheme: light){:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d23;--mut:#5b636e;--line:#e2e5ea;--zh:#2f5fd0;}}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:9px 16px;z-index:9}
  .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .grow{flex:1}
  h1{font-size:15px;margin:0;font-weight:700}
  .mut{color:var(--mut)}
  input,select,button,textarea{font:inherit;color:inherit}
  input[type=text],input[type=number],select,textarea{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 9px}
  .bar{height:6px;background:var(--line);border-radius:6px;overflow:hidden;flex:1;min-width:120px}
  .bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
  .topnav button{padding:7px 14px;border-radius:9px;border:1px solid var(--line);background:var(--card);cursor:pointer;font-weight:600}
  .topnav button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
  .topnav button:hover{border-color:var(--acc)}
  main{max-width:1120px;margin:0 auto;padding:16px}
  .sub{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:2px 0 4px}
  .subzh{font-size:14px;color:var(--zh);margin:0 0 12px;text-transform:none;letter-spacing:0}
  .pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:820px){.pair{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;position:relative}
  .card h3{margin:0 0 8px;font-size:13px;color:var(--mut);font-weight:700;letter-spacing:.03em}
  .card .txt{white-space:pre-wrap;font-size:15px}
  .card .zh{white-space:pre-wrap;font-size:14px;color:var(--zh);margin-top:10px;padding-top:10px;border-top:1px dashed var(--line)}
  .LEFT{border-top:3px solid #7aa2f7}.RIGHT{border-top:3px solid #e0af68}
  .rubric{background:var(--card);border:1px dashed var(--line);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:12.5px;color:var(--mut)}
  .rubric b{color:var(--fg)}
  /* rating axes */
  .axes{margin:14px 0 4px}
  .ax{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--line)}
  .ax:last-child{border-bottom:none}
  .ax .lab{min-width:180px;font-size:13.5px}
  .ax .lab .en{font-weight:600}.ax .lab .cn{color:var(--mut)}
  .ax .hint{color:var(--mut);font-size:12px;flex:1}
  .grp{display:flex;gap:6px}
  .grp button{padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);cursor:pointer;font-weight:600;min-width:78px;font-size:13px}
  .grp button:hover{border-color:var(--acc)}
  .grp button.selL.sel,.grp button.selR.sel{background:var(--acc);color:#fff;border-color:var(--acc)}
  .grp button.selT.sel{background:var(--warn);color:#1a1d23;border-color:var(--warn)}
  .ax.overall{background:var(--card);border:1px solid var(--acc);border-radius:10px;padding:12px 14px;margin-bottom:8px}
  .ax.overall .lab{min-width:210px}
  .ax.overall .grp button{padding:11px 16px;min-width:118px;font-size:14px}
  .dimhdr{font-size:12px;color:var(--mut);margin:10px 0 2px;text-transform:uppercase;letter-spacing:.04em}
  .conf{display:flex;gap:10px;justify-content:flex-start;align-items:center;margin:12px 0 2px;color:var(--mut);font-size:13px;flex-wrap:wrap}
  .conf label{cursor:pointer}
  textarea{width:100%;margin-top:8px;min-height:38px;resize:vertical}
  .nav{display:flex;gap:10px;justify-content:space-between;align-items:center;margin-top:14px}
  .nav button{padding:9px 16px;border-radius:9px;border:1px solid var(--line);background:var(--card);cursor:pointer;font-weight:600}
  .nav button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
  .pill{padding:2px 8px;border-radius:20px;background:var(--line);font-size:12px}
  .done{color:var(--good);font-weight:700}
  .kbd{font-family:ui-monospace,Menlo,monospace;background:var(--line);padding:1px 5px;border-radius:5px;font-size:12px}
  .export{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
  .export button{padding:8px 14px;border-radius:9px;border:1px solid var(--line);background:var(--card);cursor:pointer;font-weight:600}
  .warnbox{background:#3a2b0e;color:#ffd98a;border-radius:8px;padding:8px 12px;margin:9px 0;font-size:13px;display:none}
  @media (prefers-color-scheme: light){.warnbox{background:#fff4d6;color:#7a5b00}}
  /* completion grid */
  .panel{margin-top:22px;border-top:1px solid var(--line);padding-top:14px}
  .panel h4{margin:0 0 8px;font-size:14px}
  .stats{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mut);margin-bottom:10px}
  .stats b{color:var(--fg)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(30px,1fr));gap:5px}
  .cell{aspect-ratio:1;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--mut);font-size:11px;display:flex;align-items:center;justify-content:center;cursor:pointer;user-select:none}
  .cell:hover{border-color:var(--acc)}
  .cell.done{background:var(--good);color:#fff;border-color:var(--good)}
  .cell.partial{background:var(--warn);color:#1a1d23;border-color:var(--warn)}
  .cell.cur{outline:2px solid var(--acc);outline-offset:1px}
  .legend{display:flex;gap:14px;font-size:12px;color:var(--mut);margin-top:9px;flex-wrap:wrap}
  .legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px;border:1px solid var(--line)}
  .lg-done{background:var(--good)}.lg-part{background:var(--warn)}.lg-none{background:var(--card)}
</style>
</head>
<body>
<header>
  <div class="row">
    <h1>AgentIdeaBench · NLP 假设两两评测</h1>
    <span class="pill">__SETNAME__ · 50 pairs</span>
    <span class="mut">Annotator 标注者:</span>
    <input id="aid" type="text" placeholder="e.g. __AIDS__" size="8" autocomplete="off">
    <span class="grow"></span>
    <span class="topnav row">
      <button onclick="go(-1)">‹ Prev 上一题</button>
      <span class="mut">#<input id="jump" type="number" min="1" max="50" style="width:56px" onchange="jumpTo()"></span>
      <button class="primary" onclick="go(1)">Next 下一题 ›</button>
    </span>
  </div>
  <div class="row" style="margin-top:7px">
    <span id="prog" class="mut">0 / 50</span>
    <div class="bar"><i id="progbar"></i></div>
    <label class="mut" style="font-size:12px"><input type="checkbox" id="autoadv"> auto-advance after Overall 评完总体自动跳下一题</label>
  </div>
  <div class="warnbox" id="aidwarn">Enter your annotator ID above to start. 请先在上方填写你的标注者编号（数据按编号存在本浏览器）。</div>
</header>
<main>
  <div class="sub" id="subfield">—</div>
  <div class="subzh" id="subfieldzh"></div>
  <div class="pair">
    <div class="card LEFT"><h3>◀ LEFT · 左 — Idea A</h3><div class="txt" id="leftTxt"></div><div class="zh" id="leftZh"></div></div>
    <div class="card RIGHT"><h3>RIGHT · 右 — Idea B ▶</h3><div class="txt" id="rightTxt"></div><div class="zh" id="rightZh"></div></div>
  </div>
  <div class="rubric">
    Judge <b>which idea is the better research hypothesis</b> — overall, then optionally per dimension.
    判断<b>哪条研究假设更好</b>——先给总体判断，再（可选）逐维度评。
    维度：<b>Originality 原创性</b>（真新点子还是已有工作重组） · <b>Feasibility 可行性</b>（能否设计决定性实验） ·
    <b>Clarity 清晰度</b> · <b>Impact 影响力</b> · <b>Specificity 具体性</b>（有无具体机制/方法/指标）。
    原创性与影响力最重要。只有真的分不出高下才选 <b>Tie 平局</b>。
  </div>
  <div class="axes">
    <div class="ax overall" data-ax="overall">
      <div class="lab"><span class="en">Overall 总体</span><br><span class="cn">which is the better hypothesis?</span></div>
      <div class="grp">
        <button class="selL" data-v="L" onclick="rate('overall','L')">◀ Left 左更好</button>
        <button class="selT" data-v="T" onclick="rate('overall','T')">= Tie 平局</button>
        <button class="selR" data-v="R" onclick="rate('overall','R')">Right 右更好 ▶</button>
      </div>
      <div class="hint"><span class="kbd">←</span> / <span class="kbd">=</span> / <span class="kbd">→</span></div>
    </div>
    <div class="dimhdr">By dimension · 逐维度（可选）</div>
    <div id="dimrows"></div>
  </div>
  <div class="conf">confidence 置信度:
    <label><input type="radio" name="conf" value="low"> low 低</label>
    <label><input type="radio" name="conf" value="med" checked> medium 中</label>
    <label><input type="radio" name="conf" value="high"> high 高</label>
    <span class="grow"></span>
    <span id="status" class="mut"></span>
  </div>
  <textarea id="note" placeholder="optional note 可选备注（为什么？）"></textarea>
  <div class="nav">
    <button onclick="go(-1)">‹ Prev 上一题</button>
    <span class="mut">Enter = next 回车下一题</span>
    <button class="primary" onclick="go(1)">Next 下一题 ›</button>
  </div>
  <div class="export">
    <button onclick="exportJSON()">⬇ Export JSON</button>
    <button onclick="exportCSV()">⬇ Export CSV 导出</button>
    <span id="doneMsg" class="done"></span>
  </div>

  <div class="panel">
    <h4>Progress overview · 完成度总览</h4>
    <div class="stats" id="stats"></div>
    <div class="grid" id="grid"></div>
    <div class="legend">
      <span><i class="lg-done"></i>Overall done 已评总体</span>
      <span><i class="lg-part"></i>dims only 仅评了维度</span>
      <span><i class="lg-none"></i>not started 未开始</span>
      <span class="mut">click a cell to jump · 点格子可跳转</span>
    </div>
  </div>
</main>
<script>
const SETNAME="__SETNAME__";
const PAIRS=__DATA__;
const DIMS=[["originality","Originality","原创性"],["feasibility","Feasibility","可行性"],
            ["clarity","Clarity","清晰度"],["impact","Impact","影响力"],["specificity","Specificity","具体性"]];
const AXES=["overall","originality","feasibility","clarity","impact","specificity"];
let i=0, labels={};
function key(){return "e29_"+SETNAME+"_"+(document.getElementById('aid').value.trim()||"anon");}
function load(){ try{labels=JSON.parse(localStorage.getItem(key())||"{}")}catch(e){labels={}} }
function save(){ localStorage.setItem(key(), JSON.stringify(labels)); }
function cur(){return PAIRS[i];}
function isDone(L){return L && L.overall;}
function isPartial(L){return L && !L.overall && DIMS.some(d=>L[d[0]]);}

// build the 5 dimension rows once
(function(){
  const host=document.getElementById('dimrows');
  DIMS.forEach(d=>{
    const row=document.createElement('div'); row.className='ax'; row.dataset.ax=d[0];
    row.innerHTML='<div class="lab"><span class="en">'+d[1]+'</span> <span class="cn">'+d[2]+'</span></div>'+
      '<div class="grp">'+
      '<button class="selL" data-v="L">◀ 左</button>'+
      '<button class="selT" data-v="T">= 平</button>'+
      '<button class="selR" data-v="R">右 ▶</button></div><div class="hint"></div>';
    const btns=row.querySelectorAll('button');
    btns.forEach(b=>b.onclick=()=>rate(d[0], b.dataset.v));
    host.appendChild(row);
  });
})();

function paintAxis(ax, val){
  const row=document.querySelector('.ax[data-ax="'+ax+'"]'); if(!row) return;
  row.querySelectorAll('button').forEach(b=>b.classList.toggle('sel', b.dataset.v===val));
}
function render(){
  const p=cur(), L=labels[p.pair_id]||{};
  document.getElementById('subfield').textContent="Subfield: "+p.subfield+"   ·   "+p.pair_id+"   ("+(i+1)+"/"+PAIRS.length+")";
  document.getElementById('subfieldzh').textContent=p.subfield_zh||"";
  document.getElementById('leftTxt').textContent=p.left_text;
  document.getElementById('rightTxt').textContent=p.right_text;
  document.getElementById('leftZh').textContent=p.left_text_zh||"";
  document.getElementById('rightZh').textContent=p.right_text_zh||"";
  document.getElementById('leftZh').style.display=p.left_text_zh?"block":"none";
  document.getElementById('rightZh').style.display=p.right_text_zh?"block":"none";
  AXES.forEach(ax=>paintAxis(ax, L[ax]));
  document.querySelectorAll('input[name=conf]').forEach(r=>r.checked=(r.value===(L.conf||"med")));
  document.getElementById('note').value=L.note||"";
  document.getElementById('jump').value=i+1;
  document.getElementById('status').textContent=L.overall?("overall: "+({L:"LEFT",T:"TIE",R:"RIGHT"}[L.overall])):"overall not judged";
  updateProgress(); renderGrid();
  document.getElementById('aidwarn').style.display=document.getElementById('aid').value.trim()?"none":"block";
}
function updateProgress(){
  const done=PAIRS.filter(p=>isDone(labels[p.pair_id])).length;
  document.getElementById('prog').textContent=done+" / "+PAIRS.length;
  document.getElementById('progbar').style.width=(100*done/PAIRS.length)+"%";
  document.getElementById('doneMsg').textContent=(done===PAIRS.length)?"✓ all "+PAIRS.length+" judged — please Export CSV 全部完成，请导出":"";
  // stats: overall done, fully-dimmed, ties
  let dimsDone=0, ties=0;
  PAIRS.forEach(p=>{const L=labels[p.pair_id]||{};
    if(DIMS.every(d=>L[d[0]])) dimsDone++;
    if(L.overall==='T') ties++;});
  document.getElementById('stats').innerHTML=
    "Overall done 已评总体: <b>"+done+"/"+PAIRS.length+"</b>"+
    " &nbsp; all 5 dims done 五维全评: <b>"+dimsDone+"</b>"+
    " &nbsp; ties 平局: <b>"+ties+"</b>";
}
function renderGrid(){
  const g=document.getElementById('grid'); g.innerHTML="";
  PAIRS.forEach((p,idx)=>{
    const L=labels[p.pair_id]||{}; const c=document.createElement('div');
    c.className="cell"+(isDone(L)?" done":(isPartial(L)?" partial":""))+(idx===i?" cur":"");
    c.textContent=idx+1; c.title=p.pair_id+(L.overall?(" · "+{L:"LEFT",T:"TIE",R:"RIGHT"}[L.overall]):"");
    c.onclick=()=>{i=idx;render();window.scrollTo(0,0);};
    g.appendChild(c);
  });
}
function rate(ax, v){
  const p=cur(); const L=labels[p.pair_id]||{};
  L[ax]=(L[ax]===v)?undefined:v;              // click again to unset
  L.conf=(document.querySelector('input[name=conf]:checked')||{}).value||"med";
  L.note=document.getElementById('note').value; L.ts=Date.now();
  labels[p.pair_id]=L; save();
  paintAxis(ax, L[ax]); updateProgress(); renderGrid();
  document.getElementById('status').textContent=L.overall?("overall: "+({L:"LEFT",T:"TIE",R:"RIGHT"}[L.overall])):"overall not judged";
  if(ax==='overall' && L.overall && document.getElementById('autoadv').checked) setTimeout(()=>go(1),140);
}
function saveNote(){const p=cur(); if(labels[p.pair_id]){labels[p.pair_id].note=document.getElementById('note').value; save();}}
function go(d){ if(document.activeElement&&document.activeElement.id==='note'){document.getElementById('note').blur();}
  saveNote(); i=Math.max(0,Math.min(PAIRS.length-1,i+d)); render(); window.scrollTo(0,0);}
function jumpTo(){const v=parseInt(document.getElementById('jump').value,10);
  if(v>=1&&v<=PAIRS.length){i=v-1;render();window.scrollTo(0,0);}}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'){rate('overall','L')} else if(e.key==='ArrowRight'){rate('overall','R')}
  else if(e.key==='='){rate('overall','T')} else if(e.key==='Enter'){go(1)}
});
document.getElementById('aid').addEventListener('change',()=>{load();render();});
document.getElementById('aid').addEventListener('input',()=>{document.getElementById('aidwarn').style.display=document.getElementById('aid').value.trim()?"none":"block";});
function rows(){const aid=document.getElementById('aid').value.trim()||"anon";
  return PAIRS.map(p=>{const L=labels[p.pair_id]||{};
    return {annotator:aid,set:SETNAME,pair_id:p.pair_id,subfield:p.subfield,
      overall:L.overall||"",originality:L.originality||"",feasibility:L.feasibility||"",
      clarity:L.clarity||"",impact:L.impact||"",specificity:L.specificity||"",
      confidence:L.conf||"",note:(L.note||"").replace(/\s+/g," ").trim(),ts:L.ts||""};});}
function dl(name,txt,type){const b=new Blob([txt],{type});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=name;a.click();URL.revokeObjectURL(u);}
function exportJSON(){const aid=document.getElementById('aid').value.trim()||"anon";dl("e29_"+SETNAME+"_"+aid+".json",JSON.stringify(rows(),null,1),"application/json");}
function exportCSV(){const aid=document.getElementById('aid').value.trim()||"anon";
  const head=["annotator","set","pair_id","subfield","overall","originality","feasibility","clarity","impact","specificity","confidence","note","ts"];
  const esc=v=>'"'+String(v).replace(/"/g,'""')+'"';
  const csv=[head.join(",")].concat(rows().map(r=>head.map(h=>esc(r[h])).join(","))).join("\n");
  dl("e29_"+SETNAME+"_"+aid+".csv",csv,"text/csv");}
document.getElementById('note').addEventListener('blur',saveNote);
load();render();
</script>
</body>
</html>
"""


def build(setname, aids, pairs):
    blind = []
    for p in pairs:
        rec = dict(pair_id=p["pair_id"], subfield=p["subfield"],
                   left_text=p["left"]["text"], right_text=p["right"]["text"])
        # optional Chinese translations (added by e29_translate_ideas.py)
        if p.get("subfield_zh"):
            rec["subfield_zh"] = p["subfield_zh"]
        if p["left"].get("text_zh"):
            rec["left_text_zh"] = p["left"]["text_zh"]
        if p["right"].get("text_zh"):
            rec["right_text_zh"] = p["right"]["text_zh"]
        blind.append(rec)
    data = json.dumps(blind, ensure_ascii=False).replace("</", "<\\/")
    html = (TEMPLATE.replace("__SETNAME__", setname)
                    .replace("__AIDS__", "/".join(aids))
                    .replace("__DATA__", data))
    out = OUT / f"e29_annotation_{setname}.html"
    out.write_text(html, encoding="utf-8")
    return out, len(blind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER),
                    help="pair master JSON (default reports/e29_human_eval_pairs_nlp.json)")
    args = ap.parse_args()
    master = json.loads(Path(args.master).read_text())
    pairs = master["pairs"]
    s1 = [p for p in pairs if p.get("annot_set") == "S1"]
    s2 = [p for p in pairs if p.get("annot_set") == "S2"]
    n_zh = sum(1 for p in pairs if p["left"].get("text_zh"))
    for setname, aids, sub in [("S1", ["A1", "A2", "A3"], s1), ("S2", ["A4", "A5", "A6"], s2)]:
        out, n = build(setname, aids, sub)
        print(f"wrote {out}  ({n} blind pairs, annotators {aids})")
    print(f"Chinese translations present on {n_zh}/{len(pairs)} pairs.")
    print("\nOpen each HTML locally, one annotator at a time (enter your ID). "
          "3 annotators per set -> 3 labels per pair -> Fleiss kappa on Overall.")


if __name__ == "__main__":
    main()

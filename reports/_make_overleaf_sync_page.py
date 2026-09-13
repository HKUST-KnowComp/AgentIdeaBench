r"""Render the Overleaf sync guide as a single self-contained HTML page.

Same diff, same source of truth as _make_overleaf_sync.py -- this one is the
working surface: one row per hunk, in the order the edits must be applied
(descending Overleaf line number), each with a copy button and a tick box that
survives a reload. The prose labels below are the only hand-written part; the
line numbers and both code blocks come from difflib, so the page cannot drift
from the .tex the way a hand-written guide did.

Usage:
  /usr/bin/python3 reports/_make_overleaf_sync_page.py <overleaf.zip-or-tex> <out.html>

Read-only on the repo.
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
_mod = __import__("_make_overleaf_sync")
ROOT = Path(__file__).parent.parent
LOCAL = ROOT / "docs" / "paper" / "acl_latex.tex"

# what each hunk actually changes, keyed by the Overleaf line span it targets
WHY = {
    # --- against the 2026-09-07 Overleaf baseline (preprint + author block) ---
    "L499":      ("附录 I 发布条款", "「Both licenses ship in the repository」后面补上仓库地址"),
    "L2 之后":   ("arXiv 引擎", "\\documentclass 之前插入 3 行（含 2 行注释）钉住 \\pdfoutput=1，"
                  "让 arXiv AutoTeX 走 pdflatex；本地与 Overleaf 编译结果不变"),
    "L47 之后":  ("首页脚注", "\\maketitle 之后插入 3 行不编号脚注，写 GitHub 仓库地址；"
                  "不编号是为了不占用作者块 \\thanks 的脚注计数"),
    # --- against the 2026-09-06 baseline (the 29-hunk round, already applied) ---
    "L658":      ("附录 H 收尾", "R′/B/C 记号换成 Replay / Static / Active"),
    "L656":      ("附录 H 模型层", "同上换名，数值不变"),
    "L652–L654": ("附录 H 三条 Wilcoxon", "同上换名，数值不变"),
    "L650":      ("附录 H 均值", "R′/B/C → Replay / Static / Active"),
    "L645–L647": ("附录 H 三条定义", "分解式改用全名书写"),
    "L643":      ("附录 H Design", "去掉 (R′) 记号；Track-C telemetry → Active telemetry"),
    "L641":      ("附录 H 开头", "B≈R<C → Static≈Recall<Active"),
    "L632":      ("附录 G Reading", "B≈R<C → Static≈Recall<Active"),
    "L627–L629": ("附录 G 三条结果", "B−R / C−R / C−B 换成全名，数值不变"),
    "L623":      ("附录 G 开头", "第三条 track 由 \\textbf{R} 改称 \\textbf{recall-only}"),
    "L555":      ("Table 8 删一行", "GPT-5.3 Chat 是已列模型的非推理服务路由，不是第二个模型"),
    "L547":      ("Table 8 排版", "footnotesize → scriptsize，行距 0.96 → 1.10（63 行要塞进一页）"),
    "L545":      ("Table 8 图注", "64 → 63 paired models；roster 组成挪到正文"),
    "L541":      ("附录 J 正文", "拆成 3 行：补 roster 组成（5 frontier + 25 ladder），"
                  "能力门控独立成段，并修掉 n=58 的算术缺口"),
    "L478":      ("Table 6 图注", "Tr 列图例 BC / C → S+A / A"),
    "L461–L475": ("Table 6 表体", "15 行的 Tr 列全部换成 S+A / A"),
    "L456–L457": ("Table 6 列宽", "tabcolsep 4pt→3.4pt、hskip 1.5em→1.1em，给更宽的 S+A 让位"),
    "L447":      ("附录 E critic", "Track~B's references → the Static reference sets"),
    "L415":      ("附录 F 协议失败", "两个 backbone → 一个；删掉 kimi-k2.6 那句；36 → 30 cells"),
    "L413":      ("附录 F 利用率", "b20 由 21–84% 变 21–38%；b10 由 36–85% 变 36–68%，cap 对谁都不生效"),
    "L411":      ("附录 F 饱和", "其余四→三；|d|≤0.23→0.15；10→20 两个 gainer→一个；"
                  "spread 上界 1.41 (kimi) → 1.07 (mimo)"),
    "L409":      ("附录 F 开头", "六→五 backbone；1,079/1,080 → 899/900 rollouts；3,159 → 2,643 critic 分"),
    "L405":      ("Figure 8 图注", "六→五 backbone；补左右面板各自的画法；均值 6.56/6.56/6.57/6.60、"
                  "10→20 只动 +0.04、n=5 全部 p≥0.78；36 → 30 cells"),
    "L257":      ("§5 内容 vs 过程", "R′/B/C → Replay / Static / Active"),
    "L254":      ("§5 参考文献存在性", "Track~R → recall-only；B≈R<C 换成全名"),
    "L161":      ("§3 两个对照", "去掉 Track R / R′ 记号，改用 recall-only / replay"),
    "L155":      ("§3 设置", "六→五 backbone；p≥0.32→0.30；两个 gainer→一个；利用率 21–84% → 21–38%"),
    "L139":      ("§3 两种模式定义", "删掉 “Track~B or” / “Track~C or”"),
    "L56":       ("Figure 1 图注", "重写：加 Top / Bottom 粗体分标，删掉“最强八个模型只差 0.39”一句"),
}

ASSET_WHY = {
    "fig_turn_budget.pdf":
        "Figure 8 去掉 kimi-k2.6：左图改成 5 个 backbone 的均值 ±1 SD 带 + operating-budget 竖线",
    "fig1_static_active_cutoff.pdf": "Figure 1 重排：上下面板标注与 bootstrap 带",
    "fig2_benchmark_overview.pdf": "Figure 2 pipeline 流程图重画",
}


def assets(remote_arg):
    """Every non-.tex member of the Overleaf zip, md5-compared with docs/paper/."""
    import zipfile
    p = Path(remote_arg)
    if p.suffix != ".zip":
        return [], []
    stale, same = [], []
    with zipfile.ZipFile(p) as z:
        for n in sorted(z.namelist()):
            base = Path(n).name
            if base.endswith(".tex") or n.endswith("/"):
                continue
            local = ROOT / "docs" / "paper" / base
            if not local.exists():
                continue
            import hashlib
            rm = hashlib.md5(z.read(n)).hexdigest()
            lm = hashlib.md5(local.read_bytes()).hexdigest()
            (stale if rm != lm else same).append(base)
    return stale, same


CHECKS = [
    ("总页数", "25"), ("编译错误 / undefined reference", "0 / 0"),
    ("Float too large", "0"),
    ("Overfull \\hbox", "4（表 1/4/5/7，1.5-9.0pt，长期存在）"),
    ("首页脚注出现 GitHub 链接", "1 处"),
    ("全文 github.com/HKUST-KnowComp/AgentIdeaBench", "2 处（首页脚注 + 附录 I）"),
    ("Figure 1 / Figure 2 落在", "p1 / p3"),
    ("Figure 8 落在", "p16"), ("Table 8 落在", "p21"), ("Conclusion 落在", "p10"),
    ("作者块与单位显示", "10 位作者 · HKUST + NVAITC"),
    ("全文搜 Anonymous", "0 命中"),
    ("全文搜 Track B / Track C / Track R", "0 命中"),
    ("Table 8 数据行", "63（28 开源 + 35 闭源带 †）"),
]


def build_hunks():
    a = _mod.read_remote(sys.argv[1]).split("\n")
    b = LOCAL.read_text().split("\n")
    out = []
    for tag, i1, i2, j1, j2 in reversed(_mod.hunks(a, b)):
        if tag == "insert":                      # difflib gives i1 == i2 here
            span = f"L{i1} 之后"
            action = f"插入 {j2 - j1} 行"
        else:
            span = f"L{i1 + 1}" if i2 - i1 == 1 else f"L{i1 + 1}–L{i2}"
            action = ("删除这一行" if tag == "delete"
                      else "整行替换" if (i2 - i1 == 1 and j2 - j1 == 1)
                      else f"用 {j2 - j1} 行替换这 {i2 - i1} 行")
        where, why = WHY.get(span, ("", ""))
        out.append({
            "span": span, "where": where, "why": why, "tag": tag,
            "action": action,
            "old": "\n".join(a[i1:i2]),
            "new": ("" if tag == "delete" else "\n".join(b[j1:j2])),
        })
    return out, a, b


def main():
    hs, a, b = build_hunks()
    missing = [h["span"] for h in hs if not h["where"]]
    if missing:
        raise SystemExit(f"no label for {missing}")
    na = len(a) - (1 if a and a[-1] == "" else 0)
    nb = len(b) - (1 if b and b[-1] == "" else 0)
    ctx = {
        "n": len(hs), "hunks": hs,
        "remote": Path(sys.argv[1]).name, "na": na, "nb": nb,
        "md5a": _mod.md5("\n".join(a)), "md5b": _mod.md5(LOCAL.read_text()),
        "same": assets(sys.argv[1])[1], "stale": assets(sys.argv[1])[0],
    }
    Path(sys.argv[2]).write_text(PAGE.replace("__DATA__", json.dumps(ctx, ensure_ascii=False))
                                 .replace("__FIGS__", figs_html())
                                 .replace("__CHECKS__", checks_html()))
    print(f"wrote {sys.argv[2]}  ({len(hs)} hunks)")


def figs_html():
    stale, _ = assets(sys.argv[1])
    if not stale:
        return ('<tr><td colspan="2" style="color:var(--done)">'
                '图和样式文件与线上逐字节相同，这一步没有要传的东西。</td></tr>')
    return "".join(
        f'<tr><td><code>{html.escape(f)}</code></td>'
        f'<td>{html.escape(ASSET_WHY.get(f, "内容与线上不一致，需重传"))}</td></tr>'
        for f in stale)


def checks_html():
    return "".join(f'<tr><td>{html.escape(k)}</td><td class="want">{html.escape(v)}</td></tr>'
                   for k, v in CHECKS)


PAGE = r"""<title>Overleaf 同步 29 处</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root{
  --ground:#f5f7f9; --surface:#ffffff; --sunken:#eef1f5;
  --ink:#16202e; --muted:#5d6b7e; --faint:#8695a8;
  --line:#dce2ea; --line-soft:#e8edf3;
  --accent:#2b6cb0; --accent-soft:#e9f1f9; --accent-ink:#1d4f85;
  --old:#8d5a30; --old-soft:#f6efe8;
  --danger:#a53131; --danger-soft:#faecec;
  --done:#2c7a55; --done-soft:#e7f3ec;
  --shadow:0 1px 2px rgba(22,32,46,.05), 0 8px 24px -16px rgba(22,32,46,.30);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --ground:#0f1319; --surface:#171d25; --sunken:#131920;
  --ink:#dee5ef; --muted:#93a3b7; --faint:#6e7f94;
  --line:#28313d; --line-soft:#212933;
  --accent:#74aade; --accent-soft:#182634; --accent-ink:#9cc6ec;
  --old:#c9915f; --old-soft:#251d16;
  --danger:#dd7d7d; --danger-soft:#2a1a1a;
  --done:#63b98d; --done-soft:#152420;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --ground:#0f1319; --surface:#171d25; --sunken:#131920;
  --ink:#dee5ef; --muted:#93a3b7; --faint:#6e7f94;
  --line:#28313d; --line-soft:#212933;
  --accent:#74aade; --accent-soft:#182634; --accent-ink:#9cc6ec;
  --old:#c9915f; --old-soft:#251d16;
  --danger:#dd7d7d; --danger-soft:#2a1a1a;
  --done:#63b98d; --done-soft:#152420;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  font-size:15px; line-height:1.62;
}
code,pre,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
h1,h2,h3{font-family:"IBM Plex Serif","Songti SC",Georgia,serif; text-wrap:balance; margin:0}
a{color:var(--accent)}
.wrap{max-width:1080px; margin:0 auto; padding:0 20px 96px}

/* ---- masthead ---- */
.bar{
  position:sticky; top:0; z-index:20; background:var(--surface);
  border-bottom:1px solid var(--line); box-shadow:var(--shadow);
}
.bar-in{max-width:1080px; margin:0 auto; padding:11px 20px;
  display:flex; align-items:center; gap:16px; flex-wrap:wrap}
.bar h1{font-size:16px; font-weight:600; letter-spacing:-.01em}
.bar .sub{color:var(--faint); font-size:12px; letter-spacing:.04em; text-transform:uppercase}
.prog{margin-left:auto; display:flex; align-items:center; gap:10px}
.prog .count{font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; color:var(--muted)}
.track{width:132px; height:5px; border-radius:3px; background:var(--sunken); overflow:hidden}
.track i{display:block; height:100%; width:0; background:var(--done); transition:width .25s ease}
.reset{border:1px solid var(--line); background:transparent; color:var(--muted);
  font:inherit; font-size:12px; padding:3px 9px; border-radius:5px; cursor:pointer}
.reset:hover{border-color:var(--accent); color:var(--accent)}

/* ---- intro ---- */
header.top{padding:40px 0 24px}
header.top h2{font-size:29px; font-weight:600; letter-spacing:-.02em}
header.top p{color:var(--muted); max-width:64ch; margin:10px 0 0}
.meta{display:flex; gap:26px; flex-wrap:wrap; margin-top:18px;
  padding-top:16px; border-top:1px solid var(--line-soft); font-size:12.5px}
.meta div{display:flex; flex-direction:column; gap:2px}
.meta dt{color:var(--faint); text-transform:uppercase; letter-spacing:.06em; font-size:10.5px}
.meta dd{margin:0; font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--muted)}

.rule{
  margin:26px 0 0; padding:14px 16px 14px 18px; border-radius:0 8px 8px 0;
  background:var(--accent-soft); border-left:3px solid var(--accent); color:var(--accent-ink);
}
.rule b{font-weight:600}
.rule p{margin:0} .rule p + p{margin-top:6px}

section{margin-top:44px}
.sechead{display:flex; align-items:baseline; gap:12px; margin-bottom:14px}
.sechead .num{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--faint);
  letter-spacing:.08em}
.sechead h3{font-size:19px; font-weight:600}
.sechead .note{color:var(--faint); font-size:12.5px; margin-left:auto}

/* ---- tables ---- */
.tbl-scroll{overflow-x:auto; border:1px solid var(--line); border-radius:9px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
th,td{text-align:left; padding:9px 14px; border-bottom:1px solid var(--line-soft); vertical-align:top}
tr:last-child td{border-bottom:none}
th{font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--faint); font-weight:600}
td code{font-size:12.5px; background:var(--sunken); padding:1px 5px; border-radius:4px}
td.want{font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--done); white-space:nowrap}
.aside{color:var(--muted); font-size:13px; margin:12px 0 0}

/* ---- hunk list ---- */
ol.hunks{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:10px}
li.hunk{
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  overflow:hidden; transition:border-color .18s ease, opacity .18s ease;
}
li.hunk.done{opacity:.5; border-color:var(--line-soft)}
li.hunk.done .body{display:none}
.head{display:grid; grid-template-columns:auto 106px 1fr auto; gap:14px;
  align-items:center; padding:11px 14px; cursor:pointer}
.head:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}
.tick{appearance:none; width:19px; height:19px; margin:0; flex:none; cursor:pointer;
  border:1.5px solid var(--line); border-radius:5px; background:var(--surface); position:relative}
.tick:hover{border-color:var(--done)}
.tick:checked{background:var(--done); border-color:var(--done)}
.tick:checked::after{content:""; position:absolute; left:5.5px; top:2px; width:4px; height:9px;
  border:solid #fff; border-width:0 2px 2px 0; transform:rotate(42deg)}
.tick:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
.loc{display:flex; flex-direction:column; gap:1px}
.loc .ln{font-family:"IBM Plex Mono",monospace; font-size:15px; font-weight:600;
  font-variant-numeric:tabular-nums; letter-spacing:-.01em}
.loc .act{font-size:10.5px; color:var(--faint); letter-spacing:.02em}
.desc{min-width:0}
.desc .where{font-weight:600; font-size:13.5px}
.desc .why{color:var(--muted); font-size:12.5px; line-height:1.5}
.ord{font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--faint);
  font-variant-numeric:tabular-nums}
li.hunk.del .loc .ln{color:var(--danger)}

.body{border-top:1px solid var(--line-soft); padding:4px 14px 14px}
.block{margin-top:12px}
.blabel{display:flex; align-items:center; gap:9px; margin-bottom:5px}
.blabel span{font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; font-weight:600}
.blabel .from span{color:var(--old)}
.copy{margin-left:auto; border:1px solid var(--line); background:var(--surface); color:var(--muted);
  font:inherit; font-size:11.5px; padding:2.5px 9px; border-radius:5px; cursor:pointer}
.copy:hover{border-color:var(--accent); color:var(--accent)}
.copy.ok{border-color:var(--done); color:var(--done)}
pre{margin:0; padding:11px 13px; border-radius:7px; overflow-x:auto;
  font-size:12px; line-height:1.66; white-space:pre; tab-size:2}
body.wrap-on pre{white-space:pre-wrap; overflow-wrap:anywhere}
pre.from{background:var(--old-soft); border-left:3px solid var(--old); color:var(--ink)}
pre.to{background:var(--accent-soft); border-left:3px solid var(--accent); color:var(--ink)}
pre.gone{background:var(--danger-soft); border-left:3px solid var(--danger); color:var(--ink)}
.blabel.from span{color:var(--old)} .blabel.to span{color:var(--accent-ink)}
.blabel.gone span{color:var(--danger)}

footer{margin-top:44px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:13px}
footer b{color:var(--ink); font-weight:600}
@media (max-width:640px){
  .head{grid-template-columns:auto 1fr; row-gap:8px}
  .ord{display:none}
  .loc{flex-direction:row; align-items:baseline; gap:8px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="bar"><div class="bar-in">
  <h1>Overleaf 同步</h1>
  <span class="sub" id="date"></span>
  <div class="prog">
    <span class="count" id="count">0 / 0</span>
    <span class="track"><i id="fill"></i></span>
    <label class="reset" style="cursor:pointer; display:flex; align-items:center; gap:5px">
      <input type="checkbox" id="wrap" checked style="margin:0; accent-color:var(--accent)">长行换行</label>
    <button class="reset" id="reset" type="button">重置</button>
  </div>
</div></div>

<div class="wrap">
<header class="top">
  <h2>把线上的 <span class="mono">acl_latex.tex</span> 改成本地这一版</h2>
  <p>差异全部来自 difflib 对 Overleaf 源码的实算，不是手写的。逐处改完之后重放校验通过 md5，
     所以只要每处的「原内容」和线上逐字相同，改完就一定等于本地文件。</p>
  <div class="meta">
    <div><dt>基准（Overleaf 线上）</dt><dd id="m-a"></dd></div>
    <div><dt>目标（本地）</dt><dd id="m-b"></dd></div>
    <div><dt>差异</dt><dd id="m-n"></dd></div>
  </div>
  <div class="rule">
    <p><b>必须从大往小改。</b>行号是 Overleaf 当前文件的行号；先改行号大的，后面小行号才不会错位。
       下面的顺序已经排好，从上往下做即可。</p>
    <p><b>每处先核对「原内容」。</b>只要有一处和线上对不上，说明线上又被人改过——停下来，
       重新下载 zip 重新生成，不要硬改。</p>
  </div>
</header>

<section>
  <div class="sechead"><span class="num">STEP 00</span><h3 id="s0h">先备份</h3></div>
  <p class="aside">Overleaf → Menu → Download → Source (.zip) 存一份当回滚点。<span id="s0note"></span></p>
  <div class="tbl-scroll"><table>
    <thead><tr><th>文件（本地在 docs/paper/）</th><th>为什么变了</th></tr></thead>
    <tbody>__FIGS__</tbody>
  </table></div>
  <p class="aside">下面这些与线上 md5 逐字节相同，不用动：
     <span id="unchanged" class="mono"></span></p>
</section>

<section>
  <div class="sechead"><span class="num">STEP 01</span><h3>改 acl_latex.tex</h3>
    <span class="note">点标题行可折叠已完成项</span></div>
  <p class="aside">最省事的做法是直接把本地 <code>acl_latex.tex</code> 整份上传覆盖。
     下面这 <b id="n-inline"></b> 处是逐个改的版本，也可以当核对清单用。</p>
  <ol class="hunks" id="list"></ol>
</section>

<section>
  <div class="sechead"><span class="num">STEP 02</span><h3>Recompile 之后对一遍</h3></div>
  <div class="tbl-scroll"><table>
    <thead><tr><th>检查项</th><th>期望值</th></tr></thead>
    <tbody>__CHECKS__</tbody>
  </table></div>
  <p class="aside">Figure 8 相关的数字在正文 L155、图注 L405、附录 L409 / L411 / L413 里必须一致：
     <span class="mono">5 backbones · 899 of 900 rollouts · 2,643 critic scores · 30 cells ·
     +0.04 from 10 to 20 · all p ≥ 0.78 · utilization 21–38%</span>。这一轮没动它们。</p>
</section>

<footer>
  <p><b>仍待决定：10 页 vs 9 页。</b>当前正文 10 页。实测阈值：Figure 2 高度设 2.20in 会掉回 9 页，
     2.30in 就是 10 页。三个选项——① 保持 10 页 + 完整 pipeline 图（当前）；
     ② 压回 ≤2.20in 的扁平 Figure 2；③ 删 700–800 字正文。都是一处改动。</p>
  <p style="margin-top:8px">勾选状态存在这台浏览器本地，换设备不同步。</p>
</footer>
</div>

<script>
const D = __DATA__;
const UNCHANGED = D.same;
const KEY = "ovl-sync-" + D.md5b.slice(0, 8);

document.getElementById("date").textContent = D.n + " 处 · 从大往小";
document.getElementById("m-a").textContent = D.remote + " · " + D.na + " 行 · " + D.md5a.slice(0,12) + "…";
document.getElementById("m-b").textContent = "acl_latex.tex · " + D.nb + " 行 · " + D.md5b.slice(0,12) + "…";
document.getElementById("m-n").textContent = D.n + " 处 hunk · replay OK";
document.getElementById("n-inline").textContent = D.n;
document.getElementById("unchanged").textContent = UNCHANGED.join("、") || "（无）";
document.getElementById("s0h").textContent = D.stale.length
  ? ("先备份，再传 " + D.stale.length + " 个文件") : "先备份";
document.getElementById("s0note").textContent = D.stale.length
  ? "然后上传下面的文件覆盖同名文件（Overleaf 里上传同名文件会直接替换）。"
  : "这一轮只改 .tex，图和样式文件都不用动。";

let done = {};
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { done = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(done)); } catch (e) {} }

const esc = s => s.replace(/[&<>]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;" }[c]));

const list = document.getElementById("list");
D.hunks.forEach((h, i) => {
  const li = document.createElement("li");
  li.className = "hunk" + (h.tag === "delete" ? " del" : "");
  const gone = h.tag === "delete";
  const add = h.tag === "insert";
  li.innerHTML =
    '<div class="head" tabindex="0" role="button" aria-expanded="true">' +
      '<input class="tick" type="checkbox" aria-label="标记完成">' +
      '<span class="loc"><span class="ln">' + esc(h.span) + '</span>' +
        '<span class="act">' + esc(h.action) + '</span></span>' +
      '<span class="desc"><span class="where">' + esc(h.where) + '</span>' +
        '<div class="why">' + esc(h.why) + '</div></span>' +
      '<span class="ord">' + String(i + 1).padStart(2, "0") + ' / ' + D.n + '</span>' +
    '</div>' +
    '<div class="body">' +
      (add ? "" :
      '<div class="block"><div class="blabel ' + (gone ? "gone" : "from") + '">' +
        '<span>' + (gone ? "删掉这一行" : "线上现在是") + '</span>' +
        '<button class="copy" type="button" data-c="old">复制</button></div>' +
        '<pre class="' + (gone ? "gone" : "from") + '">' + esc(h.old) + '</pre></div>') +
      (gone ? "" :
      '<div class="block"><div class="blabel to"><span>' + (add ? "插入这几行" : "改成") + '</span>' +
        '<button class="copy" type="button" data-c="new">复制</button></div>' +
        '<pre class="to">' + esc(h.new) + '</pre></div>') +
    '</div>';

  const tick = li.querySelector(".tick");
  tick.checked = !!done[h.span];
  li.classList.toggle("done", tick.checked);
  tick.addEventListener("change", () => {
    done[h.span] = tick.checked;
    if (!tick.checked) delete done[h.span];
    li.classList.toggle("done", tick.checked);
    save(); tally();
  });

  const head = li.querySelector(".head");
  head.addEventListener("click", e => { if (e.target !== tick) toggle(); });
  head.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
  });
  function toggle() {
    const open = li.classList.toggle("done");
    tick.checked = open;
    done[h.span] = open;
    if (!open) delete done[h.span];
    head.setAttribute("aria-expanded", String(!open));
    save(); tally();
  }

  li.querySelectorAll(".copy").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const text = btn.dataset.c === "old" ? h.old : h.new;
      const flash = () => {
        btn.textContent = "已复制"; btn.classList.add("ok");
        setTimeout(() => { btn.textContent = "复制"; btn.classList.remove("ok"); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(flash, fallback);
      } else { fallback(); }
      function fallback() {
        const ta = document.createElement("textarea");
        ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); flash(); } catch (err) {
          btn.textContent = "手动复制"; setTimeout(() => btn.textContent = "复制", 1600);
        }
        document.body.removeChild(ta);
      }
    });
  });
  list.appendChild(li);
});

const wrapBox = document.getElementById("wrap");
let wrapOn = true;
try { wrapOn = localStorage.getItem(KEY + "-wrap") !== "0"; } catch (e) {}
function applyWrap() {
  wrapBox.checked = wrapOn;
  document.body.classList.toggle("wrap-on", wrapOn);
  try { localStorage.setItem(KEY + "-wrap", wrapOn ? "1" : "0"); } catch (e) {}
}
wrapBox.addEventListener("change", () => { wrapOn = wrapBox.checked; applyWrap(); });
applyWrap();

function tally() {
  const n = Object.keys(done).length;
  document.getElementById("count").textContent = n + " / " + D.n;
  document.getElementById("fill").style.width = (100 * n / D.n) + "%";
}
document.getElementById("reset").addEventListener("click", () => {
  done = {}; save();
  document.querySelectorAll("li.hunk").forEach(li => {
    li.classList.remove("done");
    li.querySelector(".tick").checked = false;
    li.querySelector(".head").setAttribute("aria-expanded", "true");
  });
  tally();
});
tally();
</script>
"""

if __name__ == "__main__":
    main()

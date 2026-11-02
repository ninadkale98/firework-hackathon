"""Local training dashboard: the SFT job (loss curves, progress, errors), DPO collection,
spend and recent errors. Serves http://127.0.0.1:8765 and refreshes itself.

    .venv-fw/bin/python scripts/dashboard.py [--job codepraxis-interviewer-sft-v1]
"""
import argparse, json, re, sqlite3, time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from dotenv import dotenv_values
from fireworks import Fireworks

ROOT = Path(__file__).resolve().parent.parent
ACCOUNT = "ninadnar98-z0upz59iv"
DPO_KEY_LIKE = "dpo:%qwen35-9b-base"
DPO_TOTAL = 2500
_cache = {"at": 0.0, "data": None}


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _metrics(url):
    """Rows of numeric metrics from the job's metrics file (JSON or JSONL)."""
    if not url:
        return []
    r = httpx.get(url, timeout=30)
    if r.status_code != 200 or r.text.lstrip().startswith("<?xml"):
        return []
    text = r.text.strip()
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("metrics") or data.get("steps") or [data]
    except ValueError:
        rows = [json.loads(l) for l in text.splitlines() if l.strip().startswith("{")]
    out = []
    for row in rows:
        flat = {}
        def walk(d, p=""):
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    walk(v, f"{p}{k}.")
                elif _num(v) is not None:
                    flat[f"{p}{k}"] = _num(v)
        if isinstance(row, dict):
            walk(row)
        if flat:
            out.append(flat)
    return out


def _render_check(url):
    if not url:
        return {"state": "not yet"}
    r = httpx.get(url, timeout=30)
    if r.status_code != 200 or "NoSuchKey" in r.text[:300]:
        return {"state": "not yet written"}
    t = r.text
    return {"state": "ready", "think_tags": t.count("<think>") + t.count("</think>"),
            "assistant_turns": t.count("<|im_start|>assistant"), "move_object": t.count("move_object"),
            "bytes": len(t)}


def sft(job):
    c = Fireworks(api_key=dotenv_values(ROOT / ".env")["FIREWORKS_API_KEY"], account_id=ACCOUNT)
    j = c.supervised_fine_tuning_jobs.get(job)
    x = {**j.model_dump(), **(j.model_extra or {})}
    prog = x.get("jobProgress") or x.get("job_progress") or {}
    return {
        "state": str(x.get("state")).replace("JOB_STATE_", ""),
        "message": (x.get("status") or {}).get("message", ""),
        "percent": prog.get("percent"), "epoch": prog.get("epoch"),
        "failed_rows": prog.get("failedRequests"), "length_capped_rows": prog.get("lengthCappedRequests"),
        "learning_rate": x.get("learningRate"), "batch_size": x.get("batchSize"),
        "lora_rank": x.get("loraRank") or x.get("lora_rank"), "lora_alpha": x.get("loraAlpha"),
        "epochs": x.get("epochs"), "created": str(x.get("createTime") or x.get("create_time")),
        "metrics": _metrics(x.get("metricsFileSignedUrl")),
        "render": _render_check(x.get("renderSamplesSignedUrl")),
    }


def dpo():
    conn = sqlite3.connect(ROOT / "data" / "distill.db")
    done = errors = 0
    reasons = Counter()
    times = []
    for doc, at in conn.execute("select doc, created_at from runs where model like ?", (DPO_KEY_LIKE,)):
        d = json.loads(doc)
        done += 1
        times.append(at)
        if d.get("error"):
            errors += 1
        elif not d.get("move"):
            reasons["invalid output"] += 1
    for (doc,) in conn.execute("select doc from scores where model like ? and kind='judge_dpo'", (DPO_KEY_LIKE,)):
        v = json.loads(doc)
        if v.get("leak_ok") is False:
            reasons["gives answer away"] += 1
        if (v.get("safety") or {}).get("ok") is False:
            reasons["fails safety"] += 1
        if (v.get("overall") or 5) <= 2:
            reasons["judge score <= 2"] += 1
    rate = None
    if len(times) > 5:
        recent = sorted(times)[-60:]
        span = recent[-1] - recent[0]
        rate = (len(recent) - 1) / span * 60 if span > 0 else None
    spend = dict(conn.execute("select provider, round(sum(usd),2) from spend group by 1").fetchall())
    pairs = sum(1 for _ in open(ROOT / "data/export/dpo.jsonl")) if (ROOT / "data/export/dpo.jsonl").exists() else 0
    return {"done": done, "total": DPO_TOTAL, "errors": errors, "reasons": dict(reasons),
            "per_min": rate, "pairs_exported": pairs, "spend": spend}


def recent_errors():
    lines = []
    for p in [ROOT / "logs/dpo/main.log", ROOT / "logs/dpo/progress.log",
              *sorted((ROOT / "fireworks-training-runs").glob("*/monitor.log"))]:
        if p.exists():
            for l in p.read_text(errors="ignore").splitlines()[-400:]:
                if re.search(r"error|traceback|exception|failed|stall|429|503", l, re.I) and "waiting (" not in l:
                    lines.append(f"{p.parent.name}/{p.name}: {l.strip()[:220]}")
    return lines[-25:]


def status(job):
    if time.time() - _cache["at"] < 15 and _cache["data"]:
        return _cache["data"]
    data = {"at": time.strftime("%H:%M:%S"), "job": job}
    for name, fn in (("sft", lambda: sft(job)), ("dpo", dpo), ("errors", recent_errors)):
        try:
            data[name] = fn()
        except Exception as e:  # one failing panel must not blank the page
            data[name] = {"panel_error": repr(e)[:300]}
    _cache.update(at=time.time(), data=data)
    return data


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Training Monitor</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--s2:#f3f2ef;--bd:#e2e0da;--ink:#0b0b0b;--i2:#52514e;--i3:#7a7974;--b:#2a78d6;--o:#eb6834;--g:#1baf7a;--r:#d03b3b;--grid:#e9e7e1}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--s2:#232321;--bd:#34332f;--ink:#f0efec;--i2:#c3c2b7;--i3:#8f8e86;--b:#3987e5;--o:#d95926;--g:#199e70;--r:#e66767;--grid:#2c2b28}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:0 auto;padding:22px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 10px}
.muted{color:var(--i3);font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:12px 14px}.tile .k{color:var(--i2);font-size:12.5px}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}.bar{height:8px;background:var(--grid);border-radius:99px;overflow:hidden;margin-top:6px}
.bar div{height:100%;background:var(--b)}.ok{color:var(--g)}.bad{color:var(--r)}.warn{color:var(--o)}
.card{border:1px solid var(--bd);border-radius:10px;padding:12px}svg text{fill:var(--i2);font-size:11px}
pre{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:12px;font:12px/1.5 ui-monospace,Menlo,monospace;overflow-x:auto;white-space:pre-wrap;margin:0}
.legend span{margin-right:14px;font-size:13px;color:var(--i2)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
</style></head><body><main>
<h1>Training Monitor</h1><p class="muted" id="at">loading…</p>
<h2>SFT job <span class="muted" id="job"></span></h2><div class="grid" id="sft"></div>
<h2>Loss <span class="muted">(training and validation, from Fireworks' metrics file)</span></h2>
<div class="card"><div class="legend" id="lg"></div><div id="chart"><p class="muted">No metrics yet: they appear once training steps run.</p></div></div>
<h2>Template check <span class="muted">(rendered training samples)</span></h2><div class="grid" id="render"></div>
<h2>DPO collection <span class="muted">(base Qwen3.5-9B answers, DeepSeek judges)</span></h2><div class="grid" id="dpo"></div>
<h2>Recent errors</h2><pre id="errs">–</pre>
</main><script>
const $=id=>document.getElementById(id), fmt=(v,d=0)=>v==null?"–":(typeof v==="number"?v.toLocaleString(undefined,{maximumFractionDigits:d}):v);
const tile=(k,v,cls="",extra="")=>`<div class="tile"><div class="k">${k}</div><div class="v ${cls}">${v}</div>${extra}</div>`;
function chart(rows){
  if(!rows||!rows.length) return;
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const lossKeys=keys.filter(k=>/loss/i.test(k)).slice(0,4);
  const xKey=keys.find(k=>/(^|\.)(step|global_step|train_step)$/i.test(k))||null;
  if(!lossKeys.length){$("chart").innerHTML=`<p class="muted">Metrics file has no loss fields yet. Fields: ${keys.slice(0,20).join(", ")}</p>`;return;}
  const W=Math.max(320,$("chart").clientWidth||900),H=260,L=48,R=12,T=10,B=28,cols=["var(--b)","var(--o)","var(--g)","var(--r)"];
  const series=lossKeys.map(k=>rows.map((r,i)=>[xKey&&r[xKey]!=null?r[xKey]:i,r[k]]).filter(p=>p[1]!=null));
  const xs=series.flat().map(p=>p[0]),ys=series.flat().map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs)||1,y0=Math.min(...ys),y1=Math.max(...ys);
  const sx=x=>L+(x-x0)/((x1-x0)||1)*(W-L-R),sy=y=>T+(1-(y-y0)/((y1-y0)||1))*(H-T-B);
  let g="";for(let i=0;i<=4;i++){const y=y0+(y1-y0)*i/4;g+=`<line x1="${L}" x2="${W-R}" y1="${sy(y)}" y2="${sy(y)}" stroke="var(--grid)"/><text x="${L-6}" y="${sy(y)+4}" text-anchor="end">${y.toFixed(3)}</text>`;}
  g+=`<text x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">${xKey||"metrics row"}</text>`;
  series.forEach((s,i)=>{g+=`<polyline fill="none" stroke="${cols[i]}" stroke-width="2" points="${s.map(p=>sx(p[0])+","+sy(p[1])).join(" ")}"/>`;const p=s[s.length-1];if(p)g+=`<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="${cols[i]}"/>`;});
  $("chart").innerHTML=`<svg width="${W}" height="${H}">${g}</svg>`;
  $("lg").innerHTML=lossKeys.map((k,i)=>{const s=series[i],last=s.length?s[s.length-1][1]:null;return `<span><span class="dot" style="background:${cols[i]}"></span>${k}: <b>${last==null?"–":last.toFixed(4)}</b></span>`}).join("");
}
async function load(){
  let d; try{d=await (await fetch("/api/status")).json();}catch(e){$("at").textContent="dashboard server not reachable";return;}
  $("at").textContent=`updated ${d.at} · refreshes every 20 s`; $("job").textContent=d.job;
  const s=d.sft||{};
  if(s.panel_error){$("sft").innerHTML=tile("SFT panel error",s.panel_error,"bad");}
  else{
    const cls=/FAIL|CANCEL/.test(s.state)?"bad":/COMPLETED/.test(s.state)?"ok":"";
    $("sft").innerHTML=tile("State",s.state,cls)+tile("Progress",fmt(s.percent)+"%","",`<div class="bar"><div style="width:${s.percent||0}%"></div></div>`)
      +tile("Epoch",`${fmt(s.epoch)} / ${fmt(s.epochs)}`)+tile("Learning rate",s.learning_rate?s.learning_rate:"platform default")
      +tile("Batch size",s.batch_size||"platform default")+tile("LoRA rank / alpha",`${fmt(s.lora_rank)} / ${s.lora_alpha||"default"}`)
      +tile("Failed rows",fmt(s.failed_rows),s.failed_rows?"bad":"ok")+tile("Too-long rows (cut)",fmt(s.length_capped_rows),s.length_capped_rows?"warn":"ok")
      +(s.message?`<div class="tile" style="grid-column:1/-1"><div class="k">Status message</div><div>${s.message}</div></div>`:"");
    chart(s.metrics);
    const r=s.render||{};
    $("render").innerHTML=r.state!=="ready"?tile("Render samples",r.state||"–"):
      tile("Thinking tags in targets",fmt(r.think_tags),r.think_tags?"bad":"ok")+tile("Assistant turns",fmt(r.assistant_turns))+tile("Targets with move_object",fmt(r.move_object),r.move_object?"ok":"warn");
  }
  const p=d.dpo||{};
  if(p.panel_error){$("dpo").innerHTML=tile("DPO panel error",p.panel_error,"bad");}
  else{
    const pct=p.total?Math.round(100*p.done/p.total):0, eta=p.per_min?Math.round((p.total-p.done)/p.per_min):null;
    $("dpo").innerHTML=tile("Prompts answered",`${fmt(p.done)} / ${fmt(p.total)}`,"",`<div class="bar"><div style="width:${pct}%"></div></div>`)
      +tile("Speed",p.per_min?fmt(p.per_min,1)+" / min":"–")+tile("ETA",eta==null?"–":eta+" min")+tile("Call errors",fmt(p.errors),p.errors?"bad":"ok")
      +Object.entries(p.reasons||{}).map(([k,v])=>tile("Rejected: "+k,fmt(v),"warn")).join("")
      +tile("Pairs exported",fmt(p.pairs_exported))
      +Object.entries(p.spend||{}).map(([k,v])=>tile("Tracked spend: "+k,"$"+fmt(v,2))).join("");
  }
  $("errs").textContent=(Array.isArray(d.errors)&&d.errors.length)?d.errors.join("\n"):(d.errors&&d.errors.panel_error)||"No errors in the logs.";
}
load();setInterval(load,20000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    job = "codepraxis-interviewer-sft-v1"

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status(self.job), default=str).encode()
            ctype = "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=Handler.job)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="e.g. the Tailscale IP, to reach it from the tailnet only")
    a = ap.parse_args()
    Handler.job = a.job
    print(f"Training monitor on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

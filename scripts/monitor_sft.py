"""Watch a managed SFT job. Logs every 3 min to fireworks-training-runs/<job>/monitor.log
and exits (so the agent wakes) on: render samples first available, a terminal state,
no progress for 30 min, or every 30 min as a status tick."""
import json, sys, time
from dotenv import dotenv_values
from fireworks import Fireworks

A = "ninadnar98-z0upz59iv"; JOB = sys.argv[1]
LOG = f"fireworks-training-runs/{JOB}/monitor.log"
c = Fireworks(api_key=dotenv_values(".env")["FIREWORKS_API_KEY"], account_id=A)
seen_render = "--render-seen" in sys.argv
start = time.time(); last_pct, last_change = None, time.time()
while True:
    j = c.supervised_fine_tuning_jobs.get(JOB).model_dump()
    prog = j.get("job_progress") or {}
    pct = prog.get("percent"); state = j.get("state")
    line = (f"{time.strftime('%H:%M:%S')} {state} pct={pct} epoch={prog.get('epoch')} "
            f"steps={prog.get('steps') or prog.get('step')} lr={j.get('learning_rate')} bs={j.get('batch_size')} "
            f"msg={(j.get('status') or {}).get('message','')[:120]}")
    open(LOG, "a").write(line + "\n"); print(line, flush=True)
    if pct != last_pct:
        last_pct, last_change = pct, time.time()
    if any(s in str(state) for s in ("COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "DELETING")):
        print("EVENT TERMINAL"); print(json.dumps({k: j.get(k) for k in ("state", "status", "output_model", "job_progress")}, default=str)); break
    rurl = (j.get("renderSamplesSignedUrl") or (c.supervised_fine_tuning_jobs.get(JOB).model_extra or {}).get("renderSamplesSignedUrl"))
    if not seen_render and rurl:
        import httpx
        rr = httpx.get(rurl, timeout=30)
        if rr.status_code == 200 and "NoSuchKey" not in rr.text[:300]:
            print("EVENT RENDER_READY"); break
    if "RUNNING" in str(state) and time.time() - last_change > 1800:
        print("EVENT STALL (no progress for 30 min)"); break
    if time.time() - start > 1800:
        print("EVENT TICK"); break
    time.sleep(180)

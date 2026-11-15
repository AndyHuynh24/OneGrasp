# ruff: noqa: E501  (the embedded HTML / JS page)
"""Live Tiger Data dashboard for the demo: open it on another device (phone, second laptop) and
watch the data arrive: the hand rows while you move (30 per second), the robot demos after the
robot runs, the storage, and the policy trained from these rows.

    python scripts/tiger_dashboard.py            # prints http://<this Mac's IP>:8000
    python scripts/tiger_dashboard.py --port 8080

One read-only database connection polls Tiger every second (slow stats every 10 s); every viewer
gets that cached state, so many phones cost the database nothing extra. The database password
never leaves this Mac. Same Wi-Fi needed (venue Wi-Fi blocks devices? use the phone's hotspot).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAST_S, SLOW_S = 1.0, 10.0

Q = {  # every panel's SQL (also shown on the page)
    "counts": "SELECT (SELECT count(*) FROM hand_frames) AS hand_rows, "
              "(SELECT count(*) FROM demo_steps) AS robot_steps, "
              "(SELECT count(*) FROM demos) AS demos, "
              "(SELECT count(*) FROM demos WHERE source = 'teleop') AS teleop, "
              "(SELECT count(*) FROM demos WHERE source <> 'teleop') AS generated",
    "session": "SELECT session_id, started_at, meta FROM sessions "
               "WHERE source = 'onetouch_live' ORDER BY started_at DESC LIMIT 1",
    "hand": "SELECT time, frame_idx, hand, gate_closed, gate_score, grasp_x, grasp_y, grasp_z "
            "FROM hand_frames WHERE session_id = %s AND time >= %s "
            "ORDER BY time DESC LIMIT 8",
    "hand_n": "SELECT count(*), sum(CASE WHEN gate_closed THEN 1 ELSE 0 END) FROM hand_frames "
              "WHERE session_id = %s AND time >= %s",
    "demos": "SELECT demo_id, source, src_demo, take, num_steps, success FROM demos "
             "ORDER BY created_at DESC, demo_id DESC LIMIT 13",
    "storage": "SELECT sum(before_compression_total_bytes)::float8, "
               "sum(after_compression_total_bytes)::float8, count(*) "
               "FROM chunk_compression_stats('demo_steps') "
               "WHERE compression_status = 'Compressed'",
    "size": "SELECT hypertable_size('demo_steps')::float8, hypertable_size('hand_frames')::float8",
    "cagg": "SELECT a.demo_id, max(a.cup_z_max) AS cup_peak, max(a.eef_speed_max) AS speed, "
            "sum(a.steps) / 30.0 AS seconds FROM demo_steps_1s a "
            "WHERE a.demo_id IN (SELECT demo_id FROM demos ORDER BY created_at DESC, "
            "demo_id DESC LIMIT 13) GROUP BY a.demo_id",
    "policy": "SELECT run_id, created_at, name, n_train_demos, n_train_steps, n_test, n_success, "
              "metrics FROM policy_runs ORDER BY created_at DESC LIMIT 1",
}  # fmt: skip


class Poller:
    def __init__(self, url: str):
        self.url = url
        self.state: dict = {"ok": False, "error": "connecting ..."}
        self.hist: deque = deque(maxlen=90)  # (t, hand_rows, robot_steps)
        self.slow: dict = {}
        self.t_slow = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.url, autocommit=True,
                               options="-c default_transaction_read_only=on")  # fmt: skip

    def _q(self, con, key, params=()):
        cur = con.cursor()
        cur.execute(Q[key], params)
        return cur.fetchall()

    def _run(self) -> None:
        con = None
        while True:
            t0 = time.time()
            try:
                if con is None or con.closed:
                    con = self._connect()
                self._poll(con)
            except Exception as ex:
                with self.lock:
                    self.state = {**self.state, "ok": False, "error": f"{type(ex).__name__}"}
                try:
                    con.close()
                except Exception:
                    pass
                con = None
            time.sleep(max(0.05, FAST_S - (time.time() - t0)))

    def _poll(self, con) -> None:
        t = time.time()
        hand_rows, steps, demos, teleop, generated = self._q(con, "counts")[0]
        self.hist.append((t, hand_rows, steps))
        rate = {"hand": 0.0, "robot": 0.0}
        old = [h for h in self.hist if t - h[0] >= 2.0]
        if old:
            h0 = old[-1]
            rate = {"hand": (hand_rows - h0[1]) / (t - h0[0]),
                    "robot": (steps - h0[2]) / (t - h0[0])}  # fmt: skip
        spark = []
        hs = list(self.hist)
        for a, b in zip(hs, hs[1:], strict=False):
            dt = max(b[0] - a[0], 1e-3)
            spark.append([round((b[1] - a[1]) / dt, 1), round((b[2] - a[2]) / dt, 1)])
        sess = self._q(con, "session")
        live = None
        if sess:
            sid, started, meta = sess[0]
            rows = self._q(con, "hand", (sid, started))
            n, n_closed = self._q(con, "hand_n", (sid, started))[0]
            live = {"take": (meta or {}).get("take"), "started": started.isoformat(),
                    "frames": int(n or 0), "closed": int(n_closed or 0),
                    "rows": [{"t": r[0].astimezone().strftime("%H:%M:%S.%f")[:-3], "frame": r[1],
                              "hand": r[2], "closed": bool(r[3]),
                              "score": None if r[4] is None else round(r[4], 2),
                              "xyz": None if r[5] is None else [round(100 * v, 1)
                                                                for v in r[5:8]]}
                             for r in rows]}  # fmt: skip
        demo_rows = [{"id": r[0], "source": r[1], "src": r[2], "take": r[3], "steps": r[4],
                      "success": bool(r[5])} for r in self._q(con, "demos")]  # fmt: skip
        if t - self.t_slow >= SLOW_S:
            self.t_slow = t
            self.slow = self._slow(con)
        with self.lock:
            self.state = {"ok": True, "error": "", "t": time.strftime("%H:%M:%S"),
                          "hand_rows": hand_rows, "robot_steps": steps, "demos": demos,
                          "teleop": teleop, "generated": generated, "rate": rate,
                          "spark": spark[-60:], "live": live, "recent": demo_rows,
                          "poll_ms": round(1e3 * (time.time() - t)), **self.slow}  # fmt: skip

    def _slow(self, con) -> dict:
        out: dict = {}
        b, a, n = self._q(con, "storage")[0]
        if b:
            out["storage"] = {"before_mb": round(b / 1e6, 1), "after_mb": round(a / 1e6, 1),
                              "ratio": round(b / max(a, 1), 1), "chunks": n}  # fmt: skip
        sz = self._q(con, "size")[0]
        out["size_mb"] = {"demo_steps": round((sz[0] or 0) / 1e6, 1),
                          "hand_frames": round((sz[1] or 0) / 1e6, 1)}  # fmt: skip
        out["cagg"] = {r[0]: {"cup_peak_cm": round(100 * (r[1] or 0), 1),
                              "speed": round(r[2] or 0, 2), "seconds": round(float(r[3]), 1)}
                       for r in self._q(con, "cagg")}  # fmt: skip
        reg = self._q(con, "policy") if self._exists(con, "policy_runs") else []
        if reg:
            r = reg[0]
            out["policy"] = {"run": r[0], "when": r[1].astimezone().strftime("%H:%M"), "name": r[2],
                             "train_demos": r[3], "train_steps": r[4], "test": r[5],
                             "success": r[6], "metrics": r[7] or {}}  # fmt: skip
        return out

    def _exists(self, con, table: str) -> bool:
        cur = con.cursor()
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
        return bool(cur.fetchone()[0])

    def json(self) -> bytes:
        with self.lock:
            return json.dumps({**self.state, "sql": Q}, default=str).encode()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OneTouch Live Data</title>
<style>
:root{--bg:#000;--panel:#0d0d0f;--line:#26262a;--text:#f5f5f5;--dim:#9a9aa0;--faint:#55555a;
--good:#7fd27a;--accent:#f5a623;--robot:#6aa7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.4 -apple-system,BlinkMacSystemFont,"Inter","Helvetica Neue",sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:18px 24px;
border-bottom:1px solid var(--line)}
.brand{letter-spacing:.32em;font-weight:600;font-size:14px}
.brand small{letter-spacing:.2em;color:var(--dim);font-weight:400;margin-left:14px}
.conn{font-size:12px;color:var(--dim);letter-spacing:.12em;text-transform:uppercase}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--faint);
margin-right:8px;vertical-align:1px}.dot.on{background:var(--good);box-shadow:0 0 10px var(--good)}
main{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;padding:20px 24px;max-width:1500px;
margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;
min-width:0}
.cap{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:10px}
.big{font-size:44px;font-weight:300;font-variant-numeric:tabular-nums;line-height:1.05}
.unit{font-size:14px;color:var(--dim);margin-left:6px}
.sub{color:var(--dim);font-size:13px;margin-top:6px;font-variant-numeric:tabular-nums}
.s3{grid-column:span 3}.s4{grid-column:span 4}.s5{grid-column:span 5}.s6{grid-column:span 6}
.s7{grid-column:span 7}.s8{grid-column:span 8}.s12{grid-column:span 12}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--dim);font-weight:500;font-size:11px;letter-spacing:.12em;text-transform:uppercase}
tr.new td{animation:flash 1.6s ease-out}@keyframes flash{from{background:#2a2410}to{background:none}}
.tag{font-size:11px;padding:2px 7px;border-radius:4px;letter-spacing:.06em}
.tag.teleop{background:#2b2312;color:var(--accent)}.tag.gen{background:#14213a;color:var(--robot)}
.held{color:var(--good)}.open{color:var(--faint)}
details{margin-top:10px}summary{color:var(--faint);font-size:11px;cursor:pointer;letter-spacing:.1em}
pre{white-space:pre-wrap;color:var(--dim);font-size:11px;margin:6px 0 0}
.bars{display:flex;align-items:flex-end;gap:6px;height:120px}
.bar{flex:1;background:var(--robot);opacity:.85;border-radius:3px 3px 0 0;min-height:2px;position:relative}
.bar.teleop{background:var(--accent)}
.legend{display:flex;gap:16px;font-size:12px;color:var(--dim);margin-top:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--dim)}
.flow b{color:var(--text);font-weight:500}.flow i{color:var(--faint);font-style:normal}
.policy .big{color:var(--good)}
svg{display:block;width:100%}
.tw{overflow-x:auto}
@media (max-width:900px){main{grid-template-columns:minmax(0,1fr);padding:14px 16px;gap:12px}
.s3,.s4,.s5,.s6,.s7,.s8,.s12{grid-column:auto}.big{font-size:36px}
header{flex-wrap:wrap;gap:6px;padding:14px 16px}.brand small{display:block;margin:4px 0 0}
.hide-m{display:none}}
</style></head><body>
<header><div class="brand">ONETOUCH<small>TIGER DATA · LIVE</small></div>
<div class="conn"><span class="dot" id="dot"></span><span id="conn">connecting</span></div></header>
<main>
<div class="card s12"><div class="flow"><b>Camera</b><i>→</i><b>3D hand</b><span>30 rows/s
into hand_frames</span><i>→</i><b>Robot demo</b><span>teleop</span><i>→</i><b>×12 generated</b>
<span>MimicGen-style</span><i>→</i><b>Tiger Data</b><span>hypertables · columnstore ·
continuous aggregates</span><i>→</i><b>Robot policy</b><span>trained with SQL</span></div></div>

<div class="card s3"><div class="cap">Hand data · live</div>
<div class="big"><span id="rh">0</span><span class="unit">rows/s</span></div>
<div class="sub"><span id="hr">–</span> hand rows stored</div></div>
<div class="card s3"><div class="cap">Robot data · live</div>
<div class="big"><span id="rr">0</span><span class="unit">steps/s</span></div>
<div class="sub"><span id="rs">–</span> robot steps stored (30 Hz)</div></div>
<div class="card s3"><div class="cap">Robot demonstrations</div>
<div class="big" id="dm">–</div>
<div class="sub"><span id="dt">–</span> led by a person → <span id="dg">–</span> generated</div></div>
<div class="card s3"><div class="cap">Columnstore</div>
<div class="big"><span id="cr">–</span><span class="unit">x smaller</span></div>
<div class="sub" id="cs">–</div></div>

<div class="card s7"><div class="cap">Ingest, last 60 s (rows per second)</div>
<svg id="spark" viewBox="0 0 600 140" preserveAspectRatio="none" height="140"></svg>
<div class="legend"><span><span class="sw" style="background:var(--accent)"></span>hand rows</span>
<span><span class="sw" style="background:var(--robot)"></span>robot steps</span></div>
<details><summary>SQL</summary><pre id="q_counts"></pre></details></div>
<div class="card s5 policy"><div class="cap">Robot policy trained from these rows</div>
<div id="pol"><div class="sub">No policy trained yet.</div></div>
<details><summary>SQL</summary><pre id="q_policy"></pre></details></div>

<div class="card s6"><div class="cap">Live take · hand_frames <span id="take"></span></div>
<div class="tw"><table><thead><tr><th>Time</th><th class="hide-m">Frame</th><th class="hide-m">Hand</th><th>Grip</th><th class="hide-m">Gate</th>
<th>Grasp xyz (cm)</th></tr></thead><tbody id="hand"></tbody></table></div>
<div class="sub" id="hs"></div>
<details><summary>SQL</summary><pre id="q_hand"></pre></details></div>
<div class="card s6"><div class="cap">Newest robot demos · demos</div>
<div class="tw"><table><thead><tr><th>Demo</th><th>Source</th><th>Steps</th><th>From</th><th class="hide-m">Take</th></tr>
</thead><tbody id="demos"></tbody></table></div>
<details><summary>SQL</summary><pre id="q_demos"></pre></details></div>

<div class="card s12"><div class="cap">Peak cup height per demo · continuous aggregate demo_steps_1s</div>
<div class="bars" id="bars"></div>
<div class="legend"><span><span class="sw" style="background:var(--accent)"></span>teleop</span>
<span><span class="sw" style="background:var(--robot)"></span>generated</span><span id="cagg_note"></span></div>
<details><summary>SQL</summary><pre id="q_cagg"></pre></details></div>
</main>
<script>
const $=id=>document.getElementById(id);const fmt=n=>n==null?'–':Number(n).toLocaleString();
let seenDemos=new Set(),seenHand=new Set(),first=true;
function spark(d){const W=600,H=140,m=Math.max(10,...d.flat());const n=Math.max(d.length-1,1);
const path=k=>d.map((v,i)=>`${i?'L':'M'}${(i/n*W).toFixed(1)},${(H-4-(v[k]/m)*(H-12)).toFixed(1)}`).join('');
$('spark').innerHTML=`<path d="${path(0)}" fill="none" stroke="#f5a623" stroke-width="2"/>`+
`<path d="${path(1)}" fill="none" stroke="#6aa7ff" stroke-width="2"/>`+
`<text x="4" y="14" fill="#55555a" font-size="11">${Math.round(m)}/s</text>`;}
async function tick(){let s;try{s=await (await fetch('api/state',{cache:'no-store'})).json()}
catch(e){$('dot').className='dot';$('conn').textContent='dashboard offline';return}
$('dot').className='dot'+(s.ok?' on':'');$('conn').textContent=s.ok?`live · ${s.t} · query ${s.poll_ms} ms`:`reconnecting (${s.error})`;
if(!s.ok)return;
$('rh').textContent=Math.round(s.rate.hand);$('rr').textContent=Math.round(s.rate.robot);
$('hr').textContent=fmt(s.hand_rows);$('rs').textContent=fmt(s.robot_steps);
$('dm').textContent=fmt(s.demos);$('dt').textContent=fmt(s.teleop);$('dg').textContent=fmt(s.generated);
if(s.storage){$('cr').textContent=s.storage.ratio;$('cs').textContent=`${s.storage.before_mb} MB stored in ${s.storage.after_mb} MB`}
spark(s.spark||[]);
const L=s.live;if(L){$('take').textContent='· take '+(L.take||'');
$('hand').innerHTML=L.rows.map(r=>{const k=r.t+r.frame,nw=!first&&!seenHand.has(k);seenHand.add(k);
return `<tr class="${nw?'new':''}"><td>${r.t}</td><td class="hide-m">${r.frame}</td><td class="hide-m">${r.hand}</td>`+
`<td class="${r.closed?'held':'open'}">${r.closed?'HOLD':'open'}</td><td class="hide-m">${r.score??'–'}</td>`+
`<td>${r.xyz?r.xyz.join(', '):'–'}</td></tr>`}).join('');
$('hs').textContent=`${fmt(L.frames)} frames this take · ${fmt(L.closed)} while holding`}
else{$('hand').innerHTML='<tr><td colspan="6" class="open">Waiting for a live take ...</td></tr>'}
$('demos').innerHTML=(s.recent||[]).map(d=>{const nw=!first&&!seenDemos.has(d.id);seenDemos.add(d.id);
const g=d.source!=='teleop';return `<tr class="${nw?'new':''}"><td>${d.id.split(':').pop()}</td>`+
`<td><span class="tag ${g?'gen':'teleop'}">${g?'generated':'teleop'}</span></td><td>${d.steps}</td>`+
`<td>${d.src?d.src.split(':').pop():'–'}</td><td class="hide-m">${d.take||''}</td></tr>`}).join('');
const C=s.cagg||{},rec=(s.recent||[]).slice().reverse(),mx=Math.max(5,...rec.map(d=>(C[d.id]||{}).cup_peak_cm||0));
$('bars').innerHTML=rec.map(d=>{const c=C[d.id];const v=c?c.cup_peak_cm:0;
return `<div class="bar ${d.source==='teleop'?'teleop':''}" style="height:${(v/mx*100).toFixed(0)}%" title="${d.id}: ${v} cm, ${c?c.seconds:'?'} s"></div>`}).join('');
$('cagg_note').textContent='refreshed every 10 s';
if(s.policy){const p=s.policy,pct=p.test?Math.round(100*p.success/p.test):0;
$('pol').innerHTML=`<div class="big">${p.success}/${p.test}<span class="unit">unseen scenes succeeded (${pct}%)</span></div>`+
`<div class="sub">${p.name} · trained ${p.when} from ${fmt(p.train_demos)} demos (${fmt(p.train_steps)} steps) read from Tiger with one SELECT</div>`+
(p.metrics&&p.metrics.place_err_cm_median!=null?`<div class="sub">median placement error ${p.metrics.place_err_cm_median} cm</div>`:'')}
for(const k of ['counts','policy','hand','demos','cagg'])$('q_'+k).textContent=s.sql[k];
first=false}
tick();setInterval(tick,1000);
</script></body></html>"""


def lan_ip() -> str:
    """This Mac's address on the local network (a UDP socket sends nothing)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--url", default=None, help="database URL (default $TIGER_DATABASE_URL)")
    args = ap.parse_args()
    url = args.url or os.environ.get("TIGER_DATABASE_URL")
    if not url:
        sys.exit("set TIGER_DATABASE_URL")
    poller = Poller(url)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/state"):
                body, ctype = poller.json(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"Tiger Data dashboard: http://{lan_ip()}:{args.port}  (this Mac: "
          f"http://localhost:{args.port})", flush=True)  # fmt: skip
    srv.serve_forever()


if __name__ == "__main__":
    main()

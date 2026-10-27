"""Keep a notebook's kernel alive so single plot cells can be re-executed in seconds.

  python nbkernel.py start  NAME NOTEBOOK     # background: run every cell once, then stay alive
  python nbkernel.py status NAME              # "running i/N" | "ready" | "dead"
  python nbkernel.py exec   NAME NOTEBOOK IDX [IDX...]   # run cell(s) from NOTEBOOK's current source
  python nbkernel.py code   NAME 'python code'           # run an ad-hoc snippet
  python nbkernel.py stop   NAME

``exec`` prints stream text and errors, and saves every image to
<STATE>/NAME/cell<IDX>_<k>.png (path printed) so you can open it with the Read tool.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

STATE = Path("/tmp/nbkernel_state")  # adjust if needed
CWD = str(Path(__file__).resolve().parents[2] / "src/pycse/examples")


def _paths(name):
    d = STATE / name
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "kernel.json", d / "status"


def _run(kc, code, timeout=3600, outdir=None, tag="out"):
    """Execute code on a kernel client; return (text, n_images, error_or_None)."""
    msg_id = kc.execute(code, store_history=True)
    text, n_img, err = [], 0, None
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            err = "client timeout"
            break
        try:
            msg = kc.get_iopub_msg(timeout=5)
        except Exception:
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, c = msg["msg_type"], msg["content"]
        if t == "stream":
            text.append(c["text"])
        elif t in ("display_data", "execute_result"):
            png = c["data"].get("image/png")
            if png and outdir is not None:
                p = Path(outdir) / f"{tag}_{n_img}.png"
                p.write_bytes(base64.b64decode(png))
                text.append(f"[image] {p}\n")
                n_img += 1
            elif "text/plain" in c["data"] and not png:
                text.append(c["data"]["text/plain"] + "\n")
        elif t == "error":
            err = "\n".join(c["traceback"])
        elif t == "status" and c["execution_state"] == "idle":
            break
    return "".join(text), n_img, err


def server(name, notebook):
    from jupyter_client.manager import start_new_kernel

    d, conn, status = _paths(name)
    nb = json.loads(Path(notebook).read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    km, kc = start_new_kernel(kernel_name="python3", cwd=CWD)
    import signal

    def _bye(*_):
        km.shutdown_kernel(now=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _bye)
    conn.write_text(Path(km.connection_file).read_text())
    (d / "pid").write_text(str(os.getpid()))
    for k, c in enumerate(cells):
        status.write_text(f"running {k + 1}/{len(cells)}")
        src = "\n".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        _, _, err = _run(kc, src, outdir=d / "init", tag=f"init{k}")
        if err:
            (d / "init_errors.log").open("a").write(f"cell(code#{k}):\n{err}\n\n")
    status.write_text("ready")
    while True:  # keep the kernel manager alive
        time.sleep(30)


def client(name):
    from jupyter_client import BlockingKernelClient

    d, conn, _ = _paths(name)
    kc = BlockingKernelClient()
    kc.load_connection_file(str(conn))
    kc.start_channels()
    return kc, d


def main():
    cmd, name = sys.argv[1], sys.argv[2]
    d, conn, status = _paths(name)
    if cmd == "server":
        (d / "init").mkdir(exist_ok=True)
        server(name, sys.argv[3])
    elif cmd == "start":
        import subprocess

        for f in (d / "init_errors.log", status):
            f.unlink(missing_ok=True)
        log = open(d / "server.log", "w")
        subprocess.Popen(
            [sys.executable, __file__, "server", name, sys.argv[3]],
            stdout=log, stderr=log, start_new_session=True,
        )  # fmt: skip
        print(f"started {name}; poll with: python nbkernel.py status {name}")
    elif cmd == "status":
        pid = (d / "pid")
        alive = False
        if pid.exists():
            try:
                os.kill(int(pid.read_text()), 0)
                alive = True
            except OSError:
                pass
        print(status.read_text() if status.exists() and alive else ("dead" if pid.exists() else "not started"))
    elif cmd == "exec":
        notebook = sys.argv[3]
        nb = json.loads(Path(notebook).read_text())
        kc, d = client(name)
        for idx in map(int, sys.argv[4:]):
            c = nb["cells"][idx]
            src = "".join(c["source"])
            (d / "fig").mkdir(exist_ok=True)
            text, n, err = _run(kc, src, outdir=d / "fig", tag=f"cell{idx}")
            print(f"===== cell {idx}: {n} image(s)")
            print(text.rstrip()[-3000:])
            if err:
                print("ERROR:\n" + err[-3000:])
    elif cmd == "code":
        kc, d = client(name)
        text, n, err = _run(kc, sys.argv[3], outdir=d / "fig", tag="adhoc")
        print(text.rstrip()[-3000:])
        if err:
            print("ERROR:\n" + err[-3000:])
    elif cmd == "stop":
        pid = d / "pid"
        if pid.exists():
            try:
                os.kill(int(pid.read_text()), 15)
            except OSError:
                pass
        print("stopped")


if __name__ == "__main__":
    main()

/* weakheat-demo frontend: vanilla JS + Canvas (spec sections 33-36).

Run sequence:
  1. validate parameters locally
  2. POST /api/nn/predict  (returns immediately)
  3. POST /api/firedrake/run (creates a real Kubernetes Job)
  4. poll GET /api/firedrake/{job_id} ~1/s until done/error
  5. both panels animate over the SAME time slider
*/
(function () {
  "use strict";

  const API = "https://" + window.WEAKHEAT_CONFIG.apiHost;
  const SHAPE = 33;
  const NT = 26;

  const $ = (id) => document.getElementById(id);
  const state = {
    nn: null, fd: null, jobId: null,
    playing: false, timer: null, httpMs: null,
  };

  // --- parameter sliders ------------------------------------------------
  const sliders = [
    ["x0", (v) => v.toFixed(2)], ["y0", (v) => v.toFixed(2)],
    ["sigma", (v) => v.toFixed(3)], ["alpha", (v) => v.toFixed(4)],
  ];
  for (const [id, fmt] of sliders) {
    $(id).addEventListener("input", () => { $(id + "v").textContent = fmt(parseFloat($(id).value)); });
  }
  for (const [id] of sliders) { $(id).dispatchEvent(new Event("input")); }

  function params() {
    return {
      x0: parseFloat($("x0").value),
      y0: parseFloat($("y0").value),
      sigma: parseFloat($("sigma").value),
      alpha: parseFloat($("alpha").value),
    };
  }

  function setStatus(msg, isError) {
    $("status").textContent = msg;
    $("status").className = "status" + (isError ? " error" : "");
  }

  // --- heatmap (fixed color range for BOTH panels; no per-frame normalize)
  function drawHeatmap(canvas, frame) {
    if (!frame) return;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(SHAPE, SHAPE);
    for (let j = 0; j < SHAPE; j++) {
      for (let i = 0; i < SHAPE; i++) {
        const v = Math.min(1, Math.max(0, frame[j * SHAPE + i]));
        const [r, g, b] = inferno(v);
        const o = (j * SHAPE + i) * 4;
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = SHAPE; off.height = SHAPE;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  // small inferno-like colormap
  function inferno(t) {
    const stops = [
      [0, [0, 0, 4]], [0.25, [87, 16, 110]], [0.5, [188, 55, 84]],
      [0.75, [249, 133, 28]], [1, [252, 255, 164]],
    ];
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) {
        const f = (t - stops[s - 1][0]) / (stops[s][0] - stops[s - 1][0]);
        return stops[s - 1][1].map((c, i) =>
          Math.round(c + f * (stops[s][1][i] - c)));
      }
    }
    return stops[stops.length - 1][1];
  }

  // --- time slider / animation ------------------------------------------
  function showTime(k) {
    k = Math.min(NT - 1, Math.max(0, k | 0));
    $("time").value = k;
    $("timeLabel").textContent = "t = " + (k * 0.01).toFixed(2);
    drawHeatmap($("nnCanvas"), state.nn && state.nn.frames[k]);
    drawHeatmap($("fdCanvas"), state.fd && state.fd.frames[k]);
  }
  $("time").addEventListener("input", () => showTime(parseInt($("time").value, 10)));

  function setPlaying(on) {
    state.playing = on;
    $("play").textContent = on ? "Pause" : "Play";
    clearInterval(state.timer);
    if (on) {
      state.timer = setInterval(() => {
        const k = (parseInt($("time").value, 10) + 1) % NT;
        showTime(k);
      }, 250);
    }
  }
  $("play").addEventListener("click", () => setPlaying(!state.playing));

  // --- API calls ---------------------------------------------------------
  async function api(path, opts) {
    const res = await fetch(API + path, opts);
    if (!res.ok) {
      let msg = "HTTP " + res.status;
      try { msg = (await res.json()).detail || msg; } catch (e) { /* keep */ }
      throw new Error(msg);
    }
    return res.json();
  }

  async function run() {
    const p = params();
    $("run").disabled = true;
    setPlaying(false);
    state.nn = null; state.fd = null; state.httpMs = null;
    $("fdRuntime").textContent = "computing...";
    $("nnRuntime").textContent = "computing...";
    ["mFd", "mNn", "mHttp", "mL2", "mSpeedup", "mJob"].forEach((id) => { $(id).textContent = "\u2014"; });
    drawHeatmap($("nnCanvas"), null); drawHeatmap($("fdCanvas"), null);

    try {
      // both requests in parallel
      const nnPromise = (async () => {
        const t0 = performance.now();
        const r = await api("/api/nn/predict", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(p),
        });
        return { r, http: performance.now() - t0 };
      })();
      const runPromise = api("/api/firedrake/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(p),
      });

      const { r: nn, http } = await nnPromise;
      state.nn = nn; state.httpMs = http;
      $("nnRuntime").textContent = nn.inference_ms.toFixed(1) + " ms";
      $("mNn").textContent = nn.inference_ms.toFixed(1) + " ms";
      $("mHttp").textContent = http.toFixed(0) + " ms";
      setStatus("Neural prediction ready \u2014 waiting for Firedrake Kubernetes Job...");
      showTime(0);

      const { job_id } = await runPromise;
      state.jobId = job_id;
      $("mJob").textContent = "weakheat-fd-" + job_id;
      await pollJob(job_id);
    } catch (e) {
      setStatus(String(e.message || e), true);
      $("run").disabled = false;
    }
  }

  async function pollJob(jobId) {
    for (;;) {
      let st;
      try {
        st = await api("/api/firedrake/" + jobId, {});
      } catch (e) {
        setStatus(String(e.message || e), true);
        $("run").disabled = false;
        return;
      }
      if (st.status === "done") {
        state.fd = st;
        $("fdRuntime").textContent = st.runtime_ms.toFixed(0) + " ms";
        $("mFd").textContent = st.runtime_ms.toFixed(0) + " ms";
        if (st.relative_l2 != null) {
          $("mL2").textContent = (100 * st.relative_l2).toFixed(2) + " %";
          if (state.nn && state.nn.inference_ms > 0) {
            $("mSpeedup").textContent = (st.runtime_ms / state.nn.inference_ms).toFixed(0) + "x";
          }
        }
        setStatus("done \u2014 both solutions ready");
        showTime(parseInt($("time").value, 10));
        $("run").disabled = false;
        return;
      }
      if (st.status === "error") {
        setStatus("Firedrake job failed: " + (st.error || "unknown error"), true);
        $("run").disabled = false;
        return;
      }
      setStatus("Firedrake Kubernetes Job " + st.status + "...");
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  $("run").addEventListener("click", run);

  // --- static test metrics in footer -------------------------------------
  api("/api/metrics")
    .then((m) => {
      if (m.test_relative_l2_median != null) {
        $("fTestL2").textContent =
          "median " + (100 * m.test_relative_l2_median).toFixed(2) + " %, " +
          "p90 " + (100 * m.test_relative_l2_p90).toFixed(2) + " %";
      }
    })
    .catch(() => { /* metrics are optional */ });
})();

"use strict";
// The shape of a glasshouse-report/1 document. Hand-written to mirror report/schema.json;
// the vitest suite renders the Python-produced fixture, so a drift between the two fails a test.
// Nothing here is computed: the browser only draws what Python wrote.
// Direction of "better" per metric; mirrors glasshouse.scorecard.HIGHER_IS_BETTER.
const HIGHER_IS_BETTER = {
    deviance: false, d2: true, gini: true, normalized_gini: true, rmse: false, mae: false, r2: true,
    mcc: true, f1: true, roc_auc: true, average_precision: true, ks: true, log_loss: false, brier: false,
};
// Formatting and small DOM helpers. Numbers are formatted by what they are, not by magnitude
// alone: rates and probabilities to 4 significant figures, big totals with separators.
function fmt(v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v))
        return "—";
    if (!Number.isFinite(v))
        return v > 0 ? "∞" : "−∞";
    if (v === 0)
        return "0";
    const abs = Math.abs(v);
    if (abs >= 1e5 || abs < 1e-4)
        return v.toExponential(digits - 1);
    if (abs >= 1000)
        return v.toLocaleString("en", { maximumFractionDigits: 0 });
    return Number(v.toPrecision(digits)).toString();
}
function fmtInt(v) {
    return v.toLocaleString("en");
}
function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class")
            node.className = v;
        else
            node.setAttribute(k, v);
    }
    for (const c of children)
        node.append(typeof c === "string" ? document.createTextNode(c) : c);
    return node;
}
function clear(node) {
    while (node.firstChild)
        node.removeChild(node.firstChild);
}
// Is `a` better than `b` on `metric`? Returns "yes" | "no" | "tie" | "-". Mirrors scorecard._verdict.
function verdict(metric, a, b) {
    const close = (x, y) => Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(x), Math.abs(y));
    if (metric === "balance") {
        const da = Math.abs(a - 1), db = Math.abs(b - 1);
        return close(da, db) ? "tie" : da < db ? "yes" : "no";
    }
    const dir = HIGHER_IS_BETTER[metric];
    if (dir === undefined)
        return "-";
    if (close(a, b))
        return "tie";
    return (a > b) === dir ? "yes" : "no";
}
// One colour per model, stable across every chart and table. Colour-blind-safe (Okabe–Ito).
const PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"];
function colourOf(models, label) {
    const i = models.indexOf(label);
    return PALETTE[(i < 0 ? models.length : i) % PALETTE.length] ?? "#000000";
}
// Chart specs from curve data. Plotly draws them when it is available; otherwise every chart
// falls back to a table of the same numbers, so the report never shows nothing.
function plotlyAvailable() {
    return typeof Plotly !== "undefined" && Plotly !== null;
}
const LAYOUT_BASE = {
    font: { family: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", size: 12, color: "#1a1a1a" },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    margin: { l: 56, r: 16, t: 36, b: 48 },
    legend: { orientation: "h", y: -0.2 },
    hovermode: "closest",
    xaxis: { gridcolor: "#e6e6e6", zerolinecolor: "#cccccc" },
    yaxis: { gridcolor: "#e6e6e6", zerolinecolor: "#cccccc" },
};
function line(x, y, name, colour, extra = {}) {
    return { type: "scatter", mode: "lines+markers", x, y, name, line: { color: colour, width: 2 }, marker: { size: 5 }, ...extra };
}
function diagonal(hi = 1, name = "reference") {
    return { type: "scatter", mode: "lines", x: [0, hi], y: [0, hi], name, line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" };
}
function lorenzSpec(curves, models) {
    return {
        title: "Lorenz — ranked by predicted risk, low to high",
        caption: "Further below the diagonal is better ranking. Gini is blind to calibration; read it next to A/E.",
        data: [diagonal(1, "random"), ...curves.map((c) => ({ ...line(c.x, c.y, `${c.label} (Gini ${fmt(c.gini, 3)})`, colourOf(models, c.label)), mode: "lines", marker: undefined }))],
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: "cumulative share of exposure", range: [0, 1] }, yaxis: { ...LAYOUT_BASE.yaxis, title: "cumulative share of outcome", range: [0, 1] } },
        table: { columns: ["model", "Gini", "points"], rows: curves.map((c) => [c.label, fmt(c.gini), c.x.length]) },
    };
}
function liftSpec(curves, models) {
    const data = [];
    for (const c of curves) {
        const col = colourOf(models, c.label);
        data.push(line(c.bin, c.predicted, `${c.label} predicted`, col));
        data.push(line(c.bin, c.actual, `${c.label} actual`, col, { line: { color: col, dash: "dot", width: 2 } }));
    }
    const rows = [];
    for (const c of curves)
        c.bin.forEach((b, i) => rows.push([c.label, b, fmt(c.weight[i]), fmt(c.predicted[i]), fmt(c.actual[i])]));
    return {
        title: "Lift — actual vs predicted by prediction bin",
        caption: "Bins of equal weight, lowest predictions first. A good model's two lines sit on top of each other and climb.",
        data,
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: "prediction bin (low → high)" }, yaxis: { ...LAYOUT_BASE.yaxis, title: "mean outcome" } },
        table: { columns: ["model", "bin", "weight", "predicted", "actual"], rows },
    };
}
function doubleLiftSpec(c, models) {
    return {
        title: `Double lift — bins by ${c.label_a} / ${c.label_b}`,
        caption: `Left: ${c.label_b} predicts more; right: ${c.label_a} predicts more. Whichever line tracks "actual" at the ends is the one to trust there.`,
        data: [
            line(c.bin, c.actual, "actual", "#000000", { line: { color: "#000000", width: 2.5 } }),
            line(c.bin, c.predicted_a, c.label_a, colourOf(models, c.label_a)),
            line(c.bin, c.predicted_b, c.label_b, colourOf(models, c.label_b)),
        ],
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: `bin of ${c.label_a} / ${c.label_b} (low → high)` }, yaxis: { ...LAYOUT_BASE.yaxis, title: "mean outcome" } },
        table: { columns: ["bin", "ratio", "weight", "actual", c.label_a, c.label_b], rows: c.bin.map((b, i) => [b, fmt(c.ratio[i]), fmt(c.weight[i]), fmt(c.actual[i]), fmt(c.predicted_a[i]), fmt(c.predicted_b[i])]) },
    };
}
function calibrationSpec(curves, models) {
    let hi = 0;
    for (const c of curves)
        for (const v of [...c.predicted, ...c.actual])
            hi = Math.max(hi, v);
    const rows = [];
    for (const c of curves)
        c.bin.forEach((b, i) => rows.push([c.label, b, fmt(c.weight[i]), fmt(c.predicted[i]), fmt(c.actual[i]), fmt(c.actual_over_expected[i], 3)]));
    return {
        title: "Calibration — actual vs predicted per bin",
        caption: "On the dashed line the number can be trusted. Above it the model under-predicts; below, it over-predicts. Hover for A/E.",
        data: [
            diagonal(hi * 1.05, "perfect"),
            ...curves.map((c) => line(c.predicted, c.actual, c.label, colourOf(models, c.label), {
                text: c.actual_over_expected.map((r, i) => `A/E ${fmt(r, 3)} · weight ${fmt(c.weight[i])}`),
                hovertemplate: "predicted %{x}<br>actual %{y}<br>%{text}<extra></extra>",
            })),
        ],
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: "mean predicted" }, yaxis: { ...LAYOUT_BASE.yaxis, title: "mean actual" } },
        table: { columns: ["model", "bin", "weight", "predicted", "actual", "A/E"], rows },
    };
}
function rocSpec(curves, models) {
    return {
        title: "ROC — true-positive rate vs false-positive rate",
        caption: "0.5 is a coin flip. Under heavy imbalance a high AUC can hide a useless precision — read the PR curve too.",
        data: [diagonal(1, "random"), ...curves.map((c) => ({ ...line(c.fpr, c.tpr, `${c.label} (AUC ${fmt(c.auc, 3)})`, colourOf(models, c.label)), mode: "lines" }))],
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: "false-positive rate", range: [0, 1] }, yaxis: { ...LAYOUT_BASE.yaxis, title: "true-positive rate", range: [0, 1] } },
        table: { columns: ["model", "AUC", "points"], rows: curves.map((c) => [c.label, fmt(c.auc), c.fpr.length]) },
    };
}
function prSpec(curves, models) {
    const base = curves[0]?.positive_rate ?? 0;
    return {
        title: "Precision–recall",
        caption: `The dashed line is the positive rate (${fmt(base, 3)}): what a random scorer achieves. Area above it is what the model adds.`,
        data: [
            { type: "scatter", mode: "lines", x: [0, 1], y: [base, base], name: "positive rate", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
            ...curves.map((c) => ({ ...line(c.recall, c.precision, `${c.label} (AP ${fmt(c.average_precision, 3)})`, colourOf(models, c.label)), mode: "lines" })),
        ],
        layout: { ...LAYOUT_BASE, xaxis: { ...LAYOUT_BASE.xaxis, title: "recall", range: [0, 1] }, yaxis: { ...LAYOUT_BASE.yaxis, title: "precision", range: [0, 1] } },
        table: { columns: ["model", "average precision", "points"], rows: curves.map((c) => [c.label, fmt(c.average_precision), c.recall.length]) },
    };
}
function aeByFeatureSpec(tables, models) {
    const first = tables[0];
    const levels = first ? first.level : [];
    const rows = [];
    for (const t of tables)
        t.level.forEach((lv, i) => rows.push([t.label, lv, fmt(t.weight[i]), fmt(t.predicted[i]), fmt(t.actual[i]), fmt(t.actual_over_expected[i], 3)]));
    return {
        title: `A/E by ${first ? first.feature : "feature"}`,
        caption: "1 is calibrated for that segment. Above 1 the model under-predicts there; below, it over-predicts. Small-weight bins are noisy.",
        data: [
            { type: "scatter", mode: "lines", x: [levels[0] ?? "", levels[levels.length - 1] ?? ""], y: [1, 1], name: "A/E = 1", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
            ...tables.map((t) => ({ type: "bar", x: t.level, y: t.actual_over_expected, name: t.label, marker: { color: colourOf(models, t.label) }, text: t.weight.map((w) => `weight ${fmt(w)}`), hovertemplate: "%{x}<br>A/E %{y:.3f}<br>%{text}<extra></extra>" })),
        ],
        layout: { ...LAYOUT_BASE, barmode: "group", xaxis: { ...LAYOUT_BASE.xaxis, title: first ? first.feature : "", type: "category" }, yaxis: { ...LAYOUT_BASE.yaxis, title: "actual / expected" } },
        table: { columns: ["model", "level", "weight", "predicted", "actual", "A/E"], rows },
    };
}
// Render a spec into `root`: Plotly if present, else the table. Always adds the caption.
function renderChart(root, spec) {
    clear(root);
    const head = el("h3", { class: "chart-title" }, [spec.title]);
    root.append(head);
    const body = el("div", { class: "chart-body" });
    root.append(body);
    if (plotlyAvailable()) {
        Plotly.newPlot(body, spec.data, { ...spec.layout, title: undefined }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
    }
    else {
        body.append(el("p", { class: "muted" }, ["Charts unavailable offline — the same numbers, as a table:"]));
        body.append(table(spec.table.columns, spec.table.rows));
    }
    root.append(el("p", { class: "caption" }, [spec.caption]));
}
function table(columns, rows) {
    const t = el("table", { class: "grid" });
    const thead = el("thead", {}, [el("tr", {}, columns.map((c) => el("th", {}, [c])))]);
    const tbody = el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v) => el("td", { class: typeof v === "number" ? "num" : "" }, [String(v)])))));
    t.append(thead, tbody);
    return t;
}
// The screens. Each takes the document and a root element and draws; state (which model,
// which curve) lives in the selectors on the page, nothing else.
const METRIC_HELP = {
    deviance: "Family deviance: did the model fit the distribution it claims? Lower is better; the naive row is the intercept-only model.",
    d2: "Deviance explained: 1 is perfect, 0 is no better than the mean. The honest 'vs naive' number for a GLM.",
    gini: "Does the model sort risk low to high? Blind to calibration — read next to balance and A/E.",
    normalized_gini: "Gini divided by the best achievable Gini; comparable across datasets.",
    balance: "Total actual / total expected. 1 means the book adds up; 3 % under on the whole book is a problem no Gini shows.",
    rmse: "Root mean squared error; dominated by big misses on heavy tails.",
    mae: "Mean absolute error; every unit of miss costs the same.",
    r2: "Coefficient of determination for plain least squares.",
    log_loss: "Proper score on the probabilities; cannot be gamed by miscalibrating.",
    brier: "Mean squared error on probabilities; read against the always-say-0 baseline (the positive rate).",
    roc_auc: "Ranking quality across all thresholds; inflated under heavy imbalance.",
    average_precision: "Area under precision–recall; the honest ranking number for rare events.",
    ks: "Largest gap between the two classes' score distributions.",
    mcc: "Matthews correlation at the threshold: high only when all four confusion cells are right.",
    f1: "Harmonic mean of precision and recall at the threshold; ignores true negatives.",
};
function overviewScreen(doc, root) {
    clear(root);
    const metrics = Object.keys(doc.scorecards[doc.models[0]].metrics);
    const primary = doc.task.primary_metric;
    // header block
    root.append(el("p", { class: "lede" }, [
        `${doc.models.length} model${doc.models.length === 1 ? "" : "s"} on ${doc.provenance.dataset} · task ${doc.task.type} (${doc.task.family}) · ${fmtInt(doc.provenance.n_rows)} rows`,
    ]));
    // the panel
    const head = el("tr", {}, [el("th", {}, ["metric"]), ...doc.models.map((m) => el("th", { class: "model", style: `color:${colourOf(doc.models, m)}` }, [m])), el("th", {}, ["naive"])]);
    const body = el("tbody");
    for (const metric of metrics) {
        const naive = doc.naive[metric] ?? NaN;
        const values = doc.models.map((m) => doc.scorecards[m].metrics[metric] ?? NaN);
        let best = -1;
        values.forEach((v, i) => { if (best < 0 || verdict(metric, v, values[best]) === "yes")
            best = i; });
        const tr = el("tr", { class: metric === primary ? "primary" : "" }, [
            el("th", { title: METRIC_HELP[metric] ?? "" }, [metric + (metric === primary ? " ★" : "")]),
            ...values.map((v, i) => {
                const vs = verdict(metric, v, naive);
                const tick = vs === "yes" ? "✓" : vs === "no" ? "✗" : "·";
                return el("td", { class: `num${i === best && doc.models.length > 1 ? " best" : ""}`, title: `vs naive: ${vs}` }, [fmt(v), el("span", { class: `tick ${vs}` }, [tick])]);
            }),
            el("td", { class: "num muted" }, [fmt(naive)]),
        ]);
        body.append(tr);
    }
    const tbl = el("table", { class: "grid panel" }, [el("thead", {}, [head]), body]);
    root.append(tbl);
    root.append(el("p", { class: "caption" }, [
        `★ primary metric by convention for a ${doc.task.type} task, not a verdict. Bold = best model on that metric. ✓ / ✗ = better / worse than the naive baseline (the weighted mean of y${doc.task.type === "binary" ? ", i.e. the class prior" : ""}). Hover a metric for what it is for.`,
    ]));
    // provenance
    const prov = el("section", { class: "provenance" }, [
        el("h3", {}, ["Provenance"]),
        el("pre", {}, [doc.provenance.describe || "(no description given)"]),
        el("p", { class: "muted" }, [
            `Split: ${doc.provenance.split ? JSON.stringify(doc.provenance.split) : "not stated"} · rows ${fmtInt(doc.provenance.n_rows)} · weight ${fmt(doc.provenance.weight_sum)} · ` +
                `residual scatter sampled to ${fmtInt(doc.provenance.sample_rows)} rows (seed ${doc.provenance.sample_seed}); every score uses all rows.`,
        ]),
    ]);
    root.append(prov);
}
function compareScreen(doc, root) {
    clear(root);
    if (doc.models.length < 2) {
        root.append(el("p", { class: "muted" }, ["Only one model: nothing to compare. The overview holds its scorecard against the naive baseline."]));
        return;
    }
    const selA = select(doc.models, doc.models[0]);
    const selB = select(doc.models, doc.models[1]);
    const controls = el("div", { class: "controls" }, ["A ", selA, " vs B ", selB]);
    const out = el("div");
    root.append(controls, out);
    const draw = () => {
        clear(out);
        const a = selA.value, b = selB.value;
        if (a === b) {
            out.append(el("p", { class: "muted" }, ["Pick two different models."]));
            return;
        }
        const cmp = doc.comparisons.find((c) => (c.a === a && c.b === b) || (c.a === b && c.b === a));
        if (!cmp) {
            out.append(el("p", { class: "muted" }, ["No comparison stored for this pair."]));
            return;
        }
        const flipped = cmp.a !== a;
        const rows = cmp.rows.map(([m, va, vb, w]) => (flipped ? [m, vb, va, w] : [m, va, vb, w]));
        const tbl = el("table", { class: "grid panel" }, [
            el("thead", {}, [el("tr", {}, [el("th", {}, ["metric"]), el("th", { style: `color:${colourOf(doc.models, a)}` }, [a]), el("th", { style: `color:${colourOf(doc.models, b)}` }, [b]), el("th", {}, ["better"])])]),
            el("tbody", {}, rows.map(([m, va, vb, w]) => el("tr", {}, [el("th", { title: METRIC_HELP[m] ?? "" }, [m]), el("td", { class: `num${w === a ? " best" : ""}` }, [fmt(va)]), el("td", { class: `num${w === b ? " best" : ""}` }, [fmt(vb)]), el("td", { class: "muted" }, [w])]))),
        ]);
        out.append(tbl);
        const dl = doc.curves.find((c) => c.kind === "double_lift" && ((c.label_a === a && c.label_b === b) || (c.label_a === b && c.label_b === a)));
        const charts = el("div", { class: "charts two" });
        const c1 = el("div", { class: "chart" }), c2 = el("div", { class: "chart" });
        charts.append(c1, c2);
        out.append(charts);
        if (dl)
            renderChart(c1, doubleLiftSpec(dl, doc.models));
        else
            c1.append(el("p", { class: "muted" }, ["Double lift not available (a model has non-positive predictions)."]));
        const cal = doc.curves.filter((c) => c.kind === "calibration" && (c.label === a || c.label === b));
        renderChart(c2, calibrationSpec(cal, doc.models));
    };
    selA.addEventListener("change", draw);
    selB.addEventListener("change", draw);
    draw();
}
function curvesScreen(doc, root) {
    clear(root);
    const kinds = Array.from(new Set(doc.curves.map((c) => c.kind))).filter((k) => k !== "double_lift");
    const features = Array.from(new Set(Object.values(doc.residuals).flatMap((r) => r.by_feature.map((t) => t.feature))));
    const hasTime = Object.values(doc.residuals).some((r) => r.over_time !== null);
    const options = [...kinds, ...features.map((f) => `ae:${f}`), ...(hasTime ? ["ae:time"] : [])];
    const labels = { lorenz: "Lorenz", lift: "Lift", calibration: "Calibration", roc: "ROC", pr: "Precision–recall" };
    const sel = el("select");
    for (const o of options)
        sel.append(el("option", { value: o }, [o.startsWith("ae:") ? `A/E by ${o.slice(3)}` : (labels[o] ?? o)]));
    const toggles = el("span", { class: "toggles" });
    const shown = new Set(doc.models);
    for (const m of doc.models) {
        const cb = el("input", { type: "checkbox", id: `show-${m}` });
        cb.checked = true;
        cb.addEventListener("change", () => { if (cb.checked)
            shown.add(m);
        else
            shown.delete(m); draw(); });
        toggles.append(el("label", { for: `show-${m}`, style: `color:${colourOf(doc.models, m)}` }, [cb, ` ${m}`]));
    }
    const controls = el("div", { class: "controls" }, ["Show ", sel, " for ", toggles]);
    const chart = el("div", { class: "chart" });
    root.append(controls, chart);
    const draw = () => {
        const k = sel.value;
        const on = (label) => shown.has(label);
        if (k.startsWith("ae:")) {
            const f = k.slice(3);
            const tables = [];
            for (const m of doc.models) {
                if (!on(m))
                    continue;
                const r = doc.residuals[m];
                if (!r)
                    continue;
                const t = f === "time" ? r.over_time : r.by_feature.find((x) => x.feature === f) ?? null;
                if (t)
                    tables.push(t);
            }
            renderChart(chart, aeByFeatureSpec(tables, doc.models));
            return;
        }
        const pick = doc.curves.filter((c) => c.kind === k && "label" in c && on(c.label));
        switch (k) {
            case "lorenz":
                renderChart(chart, lorenzSpec(pick, doc.models));
                break;
            case "lift":
                renderChart(chart, liftSpec(pick, doc.models));
                break;
            case "calibration":
                renderChart(chart, calibrationSpec(pick, doc.models));
                break;
            case "roc":
                renderChart(chart, rocSpec(pick, doc.models));
                break;
            case "pr":
                renderChart(chart, prSpec(pick, doc.models));
                break;
            default:
                clear(chart);
                chart.append(el("p", { class: "muted" }, [`No renderer for ${k}.`]));
        }
    };
    sel.addEventListener("change", draw);
    draw();
}
function select(options, value) {
    const s = el("select");
    for (const o of options)
        s.append(el("option", { value: o }, [o]));
    s.value = value;
    return s;
}
// Entry point: read the embedded JSON, build the tabs, render. Exposed as `GlasshouseReport`
// so the tests (and the template) can call it; nothing runs on load unless the template asks.
function parseReport(text) {
    const doc = JSON.parse(text);
    if (doc.schema !== "glasshouse-report/1") {
        throw new Error(`unsupported report schema ${String(doc.schema)}; this viewer reads glasshouse-report/1`);
    }
    if (!Array.isArray(doc.models) || doc.models.length === 0)
        throw new Error("report names no models");
    return doc;
}
function renderReport(doc, root) {
    clear(root);
    const screens = [
        { id: "overview", title: "Overview", draw: overviewScreen },
        { id: "compare", title: "Compare", draw: compareScreen },
        { id: "curves", title: "Curves", draw: curvesScreen },
    ];
    const header = el("header", {}, [
        el("h1", {}, [`glasshouse report · ${doc.provenance.dataset}`]),
        el("p", { class: "muted" }, [`task ${doc.task.type} · family ${doc.task.family} · schema ${doc.schema}`]),
    ]);
    const nav = el("nav", { class: "tabs", role: "tablist" });
    const panes = el("main");
    const buttons = [];
    const paneEls = [];
    screens.forEach((s, i) => {
        const btn = el("button", { type: "button", role: "tab", "aria-controls": `pane-${s.id}`, "aria-selected": i === 0 ? "true" : "false" }, [s.title]);
        const pane = el("section", { id: `pane-${s.id}`, role: "tabpanel" });
        pane.hidden = i !== 0;
        btn.addEventListener("click", () => {
            buttons.forEach((b, j) => b.setAttribute("aria-selected", j === i ? "true" : "false"));
            paneEls.forEach((p, j) => { p.hidden = j !== i; });
            if (pane.childElementCount === 0)
                s.draw(doc, pane);
        });
        buttons.push(btn);
        paneEls.push(pane);
        nav.append(btn);
        panes.append(pane);
    });
    root.append(header, nav, panes);
    screens[0].draw(doc, paneEls[0]);
    const dl = el("a", { class: "muted small", href: "#", download: `${doc.provenance.dataset}-report.json` }, ["download the JSON behind this report"]);
    dl.addEventListener("click", (e) => {
        e.preventDefault();
        const blob = new Blob([JSON.stringify(doc)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = el("a", { href: url, download: dl.getAttribute("download") ?? "report.json" });
        document.body.append(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    });
    root.append(el("footer", {}, [dl]));
}
const GlasshouseReport = { render: renderReport, parse: parseReport, version: "0.0.1" };
globalThis.GlasshouseReport = GlasshouseReport;

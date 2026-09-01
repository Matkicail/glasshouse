// The screens. Each takes the document and a root element and draws; state (which model,
// which curve) lives in the selectors on the page, nothing else.

const METRIC_HELP: Record<string, string> = {
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

function overviewScreen(doc: ReportDoc, root: HTMLElement): void {
  clear(root);
  const metrics = Object.keys(doc.scorecards[doc.models[0]!]!.metrics);
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
    const values = doc.models.map((m) => doc.scorecards[m]!.metrics[metric] ?? NaN);
    let best = -1;
    values.forEach((v, i) => { if (best < 0 || verdict(metric, v, values[best]!) === "yes") best = i; });
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

function compareScreen(doc: ReportDoc, root: HTMLElement): void {
  clear(root);
  if (doc.models.length < 2) {
    root.append(el("p", { class: "muted" }, ["Only one model: nothing to compare. The overview holds its scorecard against the naive baseline."]));
    return;
  }
  const selA = select(doc.models, doc.models[0]!);
  const selB = select(doc.models, doc.models[1]!);
  const controls = el("div", { class: "controls" }, ["A ", selA, " vs B ", selB]);
  const out = el("div");
  root.append(controls, out);
  const draw = () => {
    clear(out);
    const a = selA.value, b = selB.value;
    if (a === b) { out.append(el("p", { class: "muted" }, ["Pick two different models."])); return; }
    const cmp = doc.comparisons.find((c) => (c.a === a && c.b === b) || (c.a === b && c.b === a));
    if (!cmp) { out.append(el("p", { class: "muted" }, ["No comparison stored for this pair."])); return; }
    const flipped = cmp.a !== a;
    const rows = cmp.rows.map(([m, va, vb, w]) => (flipped ? [m, vb, va, w] : [m, va, vb, w]) as CompareRow);
    const tbl = el("table", { class: "grid panel" }, [
      el("thead", {}, [el("tr", {}, [el("th", {}, ["metric"]), el("th", { style: `color:${colourOf(doc.models, a)}` }, [a]), el("th", { style: `color:${colourOf(doc.models, b)}` }, [b]), el("th", {}, ["better"])])]),
      el("tbody", {}, rows.map(([m, va, vb, w]) => el("tr", {}, [el("th", { title: METRIC_HELP[m] ?? "" }, [m]), el("td", { class: `num${w === a ? " best" : ""}` }, [fmt(va)]), el("td", { class: `num${w === b ? " best" : ""}` }, [fmt(vb)]), el("td", { class: "muted" }, [w])]))),
    ]);
    out.append(tbl);
    const dl = doc.curves.find((c): c is DoubleLiftCurve => c.kind === "double_lift" && ((c.label_a === a && c.label_b === b) || (c.label_a === b && c.label_b === a)));
    const charts = el("div", { class: "charts two" });
    const c1 = el("div", { class: "chart" }), c2 = el("div", { class: "chart" });
    charts.append(c1, c2);
    out.append(charts);
    if (dl) renderChart(c1, doubleLiftSpec(dl, doc.models));
    else c1.append(el("p", { class: "muted" }, ["Double lift not available (a model has non-positive predictions)."]));
    const cal = doc.curves.filter((c): c is CalibrationCurve => c.kind === "calibration" && (c.label === a || c.label === b));
    renderChart(c2, calibrationSpec(cal, doc.models));
  };
  selA.addEventListener("change", draw);
  selB.addEventListener("change", draw);
  draw();
}

function curvesScreen(doc: ReportDoc, root: HTMLElement): void {
  clear(root);
  const kinds = Array.from(new Set(doc.curves.map((c) => c.kind))).filter((k) => k !== "double_lift");
  const features = Array.from(new Set(Object.values(doc.residuals).flatMap((r) => r.by_feature.map((t) => t.feature))));
  const hasTime = Object.values(doc.residuals).some((r) => r.over_time !== null);
  const options = [
    ...kinds,
    ...features.map((f) => `oneway:${f}`),
    ...features.map((f) => `ae:${f}`),
    ...(hasTime ? ["ae:time"] : []),
  ];
  const labels: Record<string, string> = { lorenz: "Lorenz", lift: "Lift", calibration: "Calibration", roc: "ROC", pr: "Precision–recall" };
  const sel = el("select");
  const optionLabel = (o: string): string => {
    if (o.startsWith("oneway:")) return `One-way: ${o.slice(7)}`;
    if (o.startsWith("ae:")) return `A/E by ${o.slice(3)}`;
    return labels[o] ?? o;
  };
  for (const o of options) sel.append(el("option", { value: o }, [optionLabel(o)]));
  const toggles = el("span", { class: "toggles" });
  const shown = new Set(doc.models);
  for (const m of doc.models) {
    const cb = el("input", { type: "checkbox", id: `show-${m}` }) as HTMLInputElement;
    cb.checked = true;
    cb.addEventListener("change", () => { if (cb.checked) shown.add(m); else shown.delete(m); draw(); });
    toggles.append(el("label", { for: `show-${m}`, style: `color:${colourOf(doc.models, m)}` }, [cb, ` ${m}`]));
  }
  const controls = el("div", { class: "controls" }, ["Show ", sel, " for ", toggles]);
  const chart = el("div", { class: "chart" });
  root.append(controls, chart);
  const draw = () => {
    const k = sel.value;
    const on = (label: string) => shown.has(label);
    if (k.startsWith("ae:") || k.startsWith("oneway:")) {
      const oneway = k.startsWith("oneway:");
      const f = oneway ? k.slice(7) : k.slice(3);
      const tables: AEByFeature[] = [];
      for (const m of doc.models) {
        if (!on(m)) continue;
        const r = doc.residuals[m];
        if (!r) continue;
        const t = f === "time" ? r.over_time : r.by_feature.find((x) => x.feature === f) ?? null;
        if (t) tables.push(t);
      }
      renderChart(chart, oneway ? onewaySpec(tables, doc.models) : aeByFeatureSpec(tables, doc.models));
      return;
    }
    const pick = doc.curves.filter((c) => c.kind === k && "label" in c && on(c.label));
    switch (k) {
      case "lorenz": renderChart(chart, lorenzSpec(pick as LorenzCurve[], doc.models)); break;
      case "lift": renderChart(chart, liftSpec(pick as LiftCurve[], doc.models)); break;
      case "calibration": renderChart(chart, calibrationSpec(pick as CalibrationCurve[], doc.models)); break;
      case "roc": renderChart(chart, rocSpec(pick as RocCurve[], doc.models)); break;
      case "pr": renderChart(chart, prSpec(pick as PrCurve[], doc.models)); break;
      default: clear(chart); chart.append(el("p", { class: "muted" }, [`No renderer for ${k}.`]));
    }
  };
  sel.addEventListener("change", draw);
  draw();
}

function residualsScreen(doc: ReportDoc, root: HTMLElement): void {
  clear(root);
  const sel = select(doc.models, doc.models[0]!);
  root.append(el("div", { class: "controls" }, ["Model ", sel]));
  const out = el("div");
  root.append(out);
  const draw = () => {
    clear(out);
    const label = sel.value;
    const r = doc.residuals[label];
    if (!r) { out.append(el("p", { class: "muted" }, ["No residuals stored for this model."])); return; }
    const stats = el("table", { class: "grid panel" }, [
      el("thead", {}, [el("tr", {}, [el("th", {}, ["residual"]), el("th", {}, ["mean"]), el("th", {}, ["std"]), el("th", {}, ["q05"]), el("th", {}, ["median"]), el("th", {}, ["q95"])])]),
      el("tbody", {}, (["deviance", "pearson"] as const).map((kind) => {
        const q = r.summary[kind];
        return el("tr", {}, [
          el("th", { title: r.definition[kind] ?? "" }, [kind]),
          el("td", { class: "num" }, [fmt(q.mean, 3)]),
          el("td", { class: "num" }, [fmt(q.std, 3)]),
          el("td", { class: "num" }, [fmt(q["q05"], 3)]),
          el("td", { class: "num" }, [fmt(q.median, 3)]),
          el("td", { class: "num" }, [fmt(q["q95"], 3)]),
        ]);
      })),
    ]);
    out.append(stats);
    out.append(el("p", { class: "caption" }, [
      "Deviance residuals are the family's own residual: sign(y − mu) times the square root of that row's weighted deviance. Hover a row for the formula.",
    ]));
    const charts = el("div", { class: "charts two" });
    const c1 = el("div", { class: "chart" }), c2 = el("div", { class: "chart" });
    charts.append(c1, c2);
    out.append(charts);
    renderChart(c1, histogramSpec(r, label, doc.models));
    renderChart(c2, scatterSpec(r, label, doc.models, doc.provenance.sample_rows));
  };
  sel.addEventListener("change", draw);
  draw();
}

function select(options: string[], value: string): HTMLSelectElement {
  const s = el("select");
  for (const o of options) s.append(el("option", { value: o }, [o]));
  s.value = value;
  return s;
}

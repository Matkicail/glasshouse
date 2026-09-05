// Chart specs from curve data. Plotly draws them when it is available; otherwise every chart
// falls back to a table of the same numbers, so the report never shows nothing.

interface PlotlyLike {
  newPlot(root: HTMLElement, data: unknown[], layout: Record<string, unknown>, config?: Record<string, unknown>): unknown;
}
declare const Plotly: PlotlyLike | undefined;

function plotlyAvailable(): boolean {
  return typeof Plotly !== "undefined" && Plotly !== null;
}

const LAYOUT_BASE: Record<string, unknown> = {
  font: { family: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", size: 12, color: "#1a1a1a" },
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  margin: { l: 56, r: 16, t: 36, b: 48 },
  legend: { orientation: "h", y: -0.2 },
  hovermode: "closest",
  xaxis: { gridcolor: "#e6e6e6", zerolinecolor: "#cccccc" },
  yaxis: { gridcolor: "#e6e6e6", zerolinecolor: "#cccccc" },
};

interface ChartSpec {
  title: string;
  caption: string; // one line on how to read it
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  table: { columns: string[]; rows: (string | number)[][] }; // the fallback
}

function line(x: number[], y: number[], name: string, colour: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { type: "scatter", mode: "lines+markers", x, y, name, line: { color: colour, width: 2 }, marker: { size: 5 }, ...extra };
}

function diagonal(hi = 1, name = "reference"): Record<string, unknown> {
  return { type: "scatter", mode: "lines", x: [0, hi], y: [0, hi], name, line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" };
}

function lorenzSpec(curves: LorenzCurve[], models: string[]): ChartSpec {
  return {
    title: "Lorenz — ranked by predicted risk, low to high",
    caption: "Further below the diagonal is better ranking. Gini is blind to calibration; read it next to A/E.",
    data: [diagonal(1, "random"), ...curves.map((c) => ({ ...line(c.x, c.y, `${c.label} (Gini ${fmt(c.gini, 3)})`, colourOf(models, c.label)), mode: "lines", marker: undefined }))],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "cumulative share of exposure", range: [0, 1] }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "cumulative share of outcome", range: [0, 1] } },
    table: { columns: ["model", "Gini", "points"], rows: curves.map((c) => [c.label, fmt(c.gini), c.x.length]) },
  };
}

function liftSpec(curves: LiftCurve[], models: string[]): ChartSpec {
  const data: Record<string, unknown>[] = [];
  for (const c of curves) {
    const col = colourOf(models, c.label);
    data.push(line(c.bin, c.predicted, `${c.label} predicted`, col));
    data.push(line(c.bin, c.actual, `${c.label} actual`, col, { line: { color: col, dash: "dot", width: 2 } }));
  }
  const rows: (string | number)[][] = [];
  for (const c of curves) c.bin.forEach((b, i) => rows.push([c.label, b, fmt(c.weight[i]), fmt(c.predicted[i]), fmt(c.actual[i])]));
  return {
    title: "Lift — actual vs predicted by prediction bin",
    caption: "Bins of equal weight, lowest predictions first. A good model's two lines sit on top of each other and climb.",
    data,
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "prediction bin (low → high)" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean outcome" } },
    table: { columns: ["model", "bin", "weight", "predicted", "actual"], rows },
  };
}

function doubleLiftSpec(c: DoubleLiftCurve, models: string[]): ChartSpec {
  return {
    title: `Double lift — bins by ${c.label_a} / ${c.label_b}`,
    caption: `Left: ${c.label_b} predicts more; right: ${c.label_a} predicts more. Whichever line tracks "actual" at the ends is the one to trust there.`,
    data: [
      line(c.bin, c.actual, "actual", "#000000", { line: { color: "#000000", width: 2.5 } }),
      line(c.bin, c.predicted_a, c.label_a, colourOf(models, c.label_a)),
      line(c.bin, c.predicted_b, c.label_b, colourOf(models, c.label_b)),
    ],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: `bin of ${c.label_a} / ${c.label_b} (low → high)` }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean outcome" } },
    table: { columns: ["bin", "ratio", "weight", "actual", c.label_a, c.label_b], rows: c.bin.map((b, i) => [b, fmt(c.ratio[i]), fmt(c.weight[i]), fmt(c.actual[i]), fmt(c.predicted_a[i]), fmt(c.predicted_b[i])]) },
  };
}

function calibrationSpec(curves: CalibrationCurve[], models: string[]): ChartSpec {
  let hi = 0;
  for (const c of curves) for (const v of [...c.predicted, ...c.actual]) hi = Math.max(hi, v);
  const rows: (string | number)[][] = [];
  for (const c of curves) c.bin.forEach((b, i) => rows.push([c.label, b, fmt(c.weight[i]), fmt(c.predicted[i]), fmt(c.actual[i]), fmt(c.actual_over_expected[i], 3)]));
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
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "mean predicted" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean actual" } },
    table: { columns: ["model", "bin", "weight", "predicted", "actual", "A/E"], rows },
  };
}

function rocSpec(curves: RocCurve[], models: string[]): ChartSpec {
  return {
    title: "ROC — true-positive rate vs false-positive rate",
    caption: "0.5 is a coin flip. Under heavy imbalance a high AUC can hide a useless precision — read the PR curve too.",
    data: [diagonal(1, "random"), ...curves.map((c) => ({ ...line(c.fpr, c.tpr, `${c.label} (AUC ${fmt(c.auc, 3)})`, colourOf(models, c.label)), mode: "lines" }))],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "false-positive rate", range: [0, 1] }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "true-positive rate", range: [0, 1] } },
    table: { columns: ["model", "AUC", "points"], rows: curves.map((c) => [c.label, fmt(c.auc), c.fpr.length]) },
  };
}

function prSpec(curves: PrCurve[], models: string[]): ChartSpec {
  const base = curves[0]?.positive_rate ?? 0;
  return {
    title: "Precision–recall",
    caption: `The dashed line is the positive rate (${fmt(base, 3)}): what a random scorer achieves. Area above it is what the model adds.`,
    data: [
      { type: "scatter", mode: "lines", x: [0, 1], y: [base, base], name: "positive rate", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
      ...curves.map((c) => ({ ...line(c.recall, c.precision, `${c.label} (AP ${fmt(c.average_precision, 3)})`, colourOf(models, c.label)), mode: "lines" })),
    ],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "recall", range: [0, 1] }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "precision", range: [0, 1] } },
    table: { columns: ["model", "average precision", "points"], rows: curves.map((c) => [c.label, fmt(c.average_precision), c.recall.length]) },
  };
}

function aeByFeatureSpec(tables: AEByFeature[], models: string[]): ChartSpec {
  const first = tables[0];
  const levels = first ? first.level : [];
  const rows: (string | number)[][] = [];
  for (const t of tables) t.level.forEach((lv, i) => rows.push([t.label, lv, fmt(t.weight[i]), fmt(t.predicted[i]), fmt(t.actual[i]), fmt(t.actual_over_expected[i], 3)]));
  return {
    title: `A/E by ${first ? first.feature : "feature"}`,
    caption: "1 is calibrated for that segment. Above 1 the model under-predicts there; below, it over-predicts. Small-weight bins are noisy.",
    data: [
      ...tables.map((t) => ({ type: "bar", x: t.level, y: t.actual_over_expected, name: t.label, marker: { color: colourOf(models, t.label) }, text: t.weight.map((w) => `weight ${fmt(w)}`), hovertemplate: "%{x}<br>A/E %{y:.3f}<br>%{text}<extra></extra>" })),
      // drawn last so the bars cannot hide it
      { type: "scatter", mode: "lines", x: [levels[0] ?? "", levels[levels.length - 1] ?? ""], y: [1, 1], name: "A/E = 1", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
    ],
    layout: { ...LAYOUT_BASE, barmode: "group", xaxis: { ...(LAYOUT_BASE.xaxis as object), title: first ? first.feature : "", type: "category" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "actual / expected" } },
    table: { columns: ["model", "level", "weight", "predicted", "actual", "A/E"], rows },
  };
}

function onewaySpec(tables: AEByFeature[], models: string[]): ChartSpec {
  const first = tables[0];
  const data: Record<string, unknown>[] = [];
  if (first) {
    // Plotly draws an overlaying axis above the base axis. The bars go on the base axis and
    // the lines on the overlay, so the lines are never hidden behind an opaque bar; the axis
    // sides are swapped so the reader still sees the outcome on the left, weight on the right.
    data.push({
      type: "bar", x: first.level, y: first.weight, name: "exposure",
      marker: { color: "#ececec" }, hovertemplate: "%{x}<br>weight %{y:.4g}<extra></extra>",
    });
    data.push({
      type: "scatter", mode: "lines+markers", x: first.level, y: first.actual, name: "actual", yaxis: "y2",
      line: { color: "#000000", width: 2.5 }, marker: { size: 5 },
    });
  }
  for (const t of tables) {
    data.push({
      type: "scatter", mode: "lines+markers", x: t.level, y: t.predicted, name: t.label, yaxis: "y2",
      line: { color: colourOf(models, t.label), width: 2 }, marker: { size: 5 },
    });
  }
  const rows: (string | number)[][] = [];
  for (const t of tables) t.level.forEach((lv, i) => rows.push([t.label, lv, fmt(t.weight[i]), fmt(t.actual[i]), fmt(t.predicted[i])]));
  return {
    title: `One-way: ${first ? first.feature : "feature"}`,
    caption: "The black line is the actual mean per bin, the grey bars are the weight there. A model's line should track the black one where the bars are tall; thin bins are noise.",
    data,
    layout: {
      ...LAYOUT_BASE,
      xaxis: { ...(LAYOUT_BASE.xaxis as object), title: first ? first.feature : "", type: "category" },
      yaxis: { side: "right", showgrid: false, title: "weight", rangemode: "tozero" },
      yaxis2: { ...(LAYOUT_BASE.yaxis as object), overlaying: "y", side: "left", title: "mean outcome" },
      legend: { orientation: "h", y: -0.25 },
    },
    table: { columns: ["model", "level", "weight", "actual", "predicted"], rows },
  };
}

function importanceSpec(explain: Record<string, ExplainDoc>, models: string[]): ChartSpec {
  const labels = models.filter((m) => explain[m]);
  const features = Array.from(new Set(labels.flatMap((m) => explain[m]!.importance.features)));
  const rows: (string | number)[][] = [];
  for (const m of labels) explain[m]!.importance.features.forEach((f, i) => rows.push([m, f, fmt(explain[m]!.importance.mean[i]), fmt(explain[m]!.importance.std[i])]));
  return {
    title: "Permutation importance: deviance increase when a feature is shuffled",
    caption: "On held-out rows, one feature at a time is shuffled and the mean deviance re-scored; the bar is the increase, averaged over folds, the whisker its spread. A feature the model does not use costs nothing to shuffle. Compare models on the same feature, not features on absolute scale.",
    data: labels.map((m) => {
      const e = explain[m]!.importance;
      const by = new Map(e.features.map((f, i) => [f, i] as const));
      return {
        type: "bar", name: m, x: features,
        y: features.map((f) => (by.has(f) ? e.mean[by.get(f)!] : null)),
        error_y: { type: "data", array: features.map((f) => (by.has(f) ? e.std[by.get(f)!] : 0)), visible: true, thickness: 1 },
        marker: { color: colourOf(models, m) },
        hovertemplate: "%{x}<br>+%{y:.4g} deviance<extra>" + m + "</extra>",
      };
    }),
    layout: { ...LAYOUT_BASE, barmode: "group", xaxis: { ...(LAYOUT_BASE.xaxis as object), type: "category" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean deviance increase", rangemode: "tozero" } },
    table: { columns: ["model", "feature", "increase", "fold spread"], rows },
  };
}

function partialDependenceSpec(curves: { label: string; pd: PartialDependenceDoc }[], models: string[]): ChartSpec {
  const first = curves[0];
  const feature = first ? first.pd.feature : "feature";
  const categorical = first ? first.pd.kind === "categorical" : false;
  const data: Record<string, unknown>[] = [];
  const rows: (string | number)[][] = [];
  for (const { label, pd } of curves) {
    const colour = colourOf(models, label);
    pd.grid.forEach((g, i) => rows.push([label, String(g), fmt(pd.mean[i]), fmt(pd.low[i]), fmt(pd.high[i])]));
    if (categorical) {
      data.push({ type: "bar", name: label, x: pd.grid, y: pd.mean, marker: { color: colour }, error_y: { type: "data", symmetric: false, array: pd.high.map((h, i) => h - (pd.mean[i] ?? 0)), arrayminus: pd.mean.map((m, i) => m - (pd.low[i] ?? 0)), visible: true, thickness: 1 } });
    } else {
      // the fold band first (two traces, filled between), then the mean line on top
      data.push({ type: "scatter", mode: "lines", x: pd.grid, y: pd.high, line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
      data.push({ type: "scatter", mode: "lines", x: pd.grid, y: pd.low, line: { width: 0 }, fill: "tonexty", fillcolor: colour + "22", showlegend: false, hoverinfo: "skip" });
      data.push({ type: "scatter", mode: "lines+markers", x: pd.grid, y: pd.mean, name: label, line: { color: colour, width: 2 }, marker: { size: 4 } });
    }
  }
  return {
    title: `Partial dependence: ${feature}`,
    caption: "The feature is set to each grid value on every held-out row and the predictions averaged: what the model says the feature does, averaged over how the other features co-occur. The band is the spread across folds; a wide band is a model that is not sure. Points sit at the feature's quantiles, so the picture is drawn where the data is.",
    data,
    layout: { ...LAYOUT_BASE, barmode: "group", xaxis: { ...(LAYOUT_BASE.xaxis as object), title: feature, ...(categorical ? { type: "category" } : {}) }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean prediction" } },
    table: { columns: ["model", feature, "mean", "fold low", "fold high"], rows },
  };
}

function histogramSpec(r: ResidualDoc, label: string, models: string[]): ChartSpec {
  const edges = r.histogram.edges;
  const centers = r.histogram.counts.map((_, i) => ((edges[i] ?? 0) + (edges[i + 1] ?? 0)) / 2);
  return {
    title: "Deviance residuals",
    caption: "For a well-specified family this is roughly symmetric around zero with unit spread. A heavy shoulder or a shifted centre says the family or the mean model is off.",
    data: [{ type: "bar", x: centers, y: r.histogram.counts, name: label, marker: { color: colourOf(models, label) }, hovertemplate: "residual %{x:.3f}<br>rows %{y}<extra></extra>" }],
    layout: { ...LAYOUT_BASE, bargap: 0.05, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "deviance residual" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "rows" } },
    table: { columns: ["bin centre", "rows"], rows: centers.map((c, i) => [fmt(c), r.histogram.counts[i] ?? 0]) },
  };
}

function scatterSpec(r: ResidualDoc, label: string, models: string[], sampleRows: number): ChartSpec {
  return {
    title: "Residual vs fitted",
    caption: `A structureless cloud around zero is what a right model leaves behind; a funnel or a bend is a lead. Sampled to ${sampleRows.toLocaleString("en")} rows for drawing only.`,
    data: [
      { type: "scatter", mode: "lines", x: [Math.min(...r.scatter.fitted), Math.max(...r.scatter.fitted)], y: [0, 0], name: "zero", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
      { type: "scattergl", mode: "markers", x: r.scatter.fitted, y: r.scatter.deviance, name: label, marker: { color: colourOf(models, label), size: 3, opacity: 0.35 }, hovertemplate: "fitted %{x:.4g}<br>residual %{y:.3f}<extra></extra>" },
    ],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "fitted mean" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "deviance residual" } },
    table: { columns: ["shown"], rows: [[`${r.scatter.fitted.length} sampled points (open with Plotly to see them)`]] },
  };
}

// Render a spec into `root`: Plotly if present, else the table. Always adds the caption.
function renderChart(root: HTMLElement, spec: ChartSpec): void {
  clear(root);
  const head = el("h3", { class: "chart-title" }, [spec.title]);
  root.append(head);
  const body = el("div", { class: "chart-body" });
  root.append(body);
  if (plotlyAvailable()) {
    Plotly!.newPlot(body, spec.data, { ...spec.layout, title: undefined }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
  } else {
    body.append(el("p", { class: "muted" }, ["Charts unavailable offline — the same numbers, as a table:"]));
    body.append(table(spec.table.columns, spec.table.rows));
  }
  root.append(el("p", { class: "caption" }, [spec.caption]));
}

function table(columns: string[], rows: (string | number)[][]): HTMLTableElement {
  const t = el("table", { class: "grid" });
  const thead = el("thead", {}, [el("tr", {}, columns.map((c) => el("th", {}, [c])))]);
  const tbody = el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v) => el("td", { class: typeof v === "number" ? "num" : "" }, [String(v)])))));
  t.append(thead, tbody);
  return t;
}

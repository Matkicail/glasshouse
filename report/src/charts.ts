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
    title: "Lorenz curve",
    caption: "Ranked by predicted risk, low to high. Further below the diagonal is better ranking. Gini says nothing about calibration, so read it next to A/E.",
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
    title: "Lift",
    caption: "Bins of equal weight, lowest predictions first. A good model's two lines sit on top of each other and climb.",
    data,
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "prediction bin (low → high)" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "mean outcome" } },
    table: { columns: ["model", "bin", "weight", "predicted", "actual"], rows },
  };
}

function doubleLiftSpec(c: DoubleLiftCurve, models: string[]): ChartSpec {
  return {
    title: `Double lift: ${c.label_a} vs ${c.label_b}`,
    caption: `Rows are binned by ${c.label_a} / ${c.label_b}. On the left ${c.label_b} predicts more, on the right ${c.label_a} does. The line that tracks the actual line at the ends is the one to trust there.`,
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
    title: "Calibration",
    caption: "On the dashed line the number can be trusted. Above it the model under-predicts, below it the model over-predicts. Hover a point for A/E.",
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
    title: "ROC",
    caption: "True-positive rate against false-positive rate. 0.5 is a coin flip. Under heavy imbalance a high AUC can hide a useless precision, so read the PR curve too.",
    data: [diagonal(1, "random"), ...curves.map((c) => ({ ...line(c.fpr, c.tpr, `${c.label} (AUC ${fmt(c.auc, 3)})`, colourOf(models, c.label)), mode: "lines" }))],
    layout: { ...LAYOUT_BASE, xaxis: { ...(LAYOUT_BASE.xaxis as object), title: "false-positive rate", range: [0, 1] }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "true-positive rate", range: [0, 1] } },
    table: { columns: ["model", "AUC", "points"], rows: curves.map((c) => [c.label, fmt(c.auc), c.fpr.length]) },
  };
}

function prSpec(curves: PrCurve[], models: string[]): ChartSpec {
  const base = curves[0]?.positive_rate ?? 0;
  return {
    title: "Precision and recall",
    caption: `The dashed line is the positive rate (${fmt(base, 3)}), which is what a random scorer achieves. Everything above it is what the model adds.`,
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
    caption: "1 means the model is right for that segment. Above 1 it under-predicts there, below 1 it over-predicts. Bins with little weight are noisy.",
    data: [
      { type: "scatter", mode: "lines", x: [levels[0] ?? "", levels[levels.length - 1] ?? ""], y: [1, 1], name: "A/E = 1", line: { color: "#999999", dash: "dash", width: 1 }, hoverinfo: "skip" },
      ...tables.map((t) => ({ type: "bar", x: t.level, y: t.actual_over_expected, name: t.label, marker: { color: colourOf(models, t.label) }, text: t.weight.map((w) => `weight ${fmt(w)}`), hovertemplate: "%{x}<br>A/E %{y:.3f}<br>%{text}<extra></extra>" })),
    ],
    layout: { ...LAYOUT_BASE, barmode: "group", xaxis: { ...(LAYOUT_BASE.xaxis as object), title: first ? first.feature : "", type: "category" }, yaxis: { ...(LAYOUT_BASE.yaxis as object), title: "actual / expected" } },
    table: { columns: ["model", "level", "weight", "predicted", "actual", "A/E"], rows },
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
    body.append(el("p", { class: "muted" }, ["Charts are unavailable offline. The same numbers, as a table:"]));
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

// Entry point: read the embedded JSON, build the tabs, render. Exposed as `GlasshouseReport`
// so the tests (and the template) can call it; nothing runs on load unless the template asks.

interface GlasshouseReportApi {
  render(doc: ReportDoc, root: HTMLElement): void;
  parse(text: string): ReportDoc;
  version: string;
}

function parseReport(text: string): ReportDoc {
  const doc = JSON.parse(text) as Partial<ReportDoc>;
  if (doc.schema !== "glasshouse-report/1") {
    throw new Error(`unsupported report schema ${String(doc.schema)}; this viewer reads glasshouse-report/1`);
  }
  if (!Array.isArray(doc.models) || doc.models.length === 0) throw new Error("report names no models");
  return doc as ReportDoc;
}

function renderReport(doc: ReportDoc, root: HTMLElement): void {
  clear(root);
  const screens: { id: string; title: string; draw: (d: ReportDoc, r: HTMLElement) => void }[] = [
    { id: "overview", title: "Overview", draw: overviewScreen },
    { id: "compare", title: "Compare", draw: compareScreen },
    { id: "curves", title: "Curves", draw: curvesScreen },
    { id: "residuals", title: "Residuals", draw: residualsScreen },
  ];
  const header = el("header", {}, [
    el("h1", {}, [`glasshouse report · ${doc.provenance.dataset}`]),
    el("p", { class: "muted" }, [`task ${doc.task.type} · family ${doc.task.family} · schema ${doc.schema}`]),
  ]);
  const nav = el("nav", { class: "tabs", role: "tablist" });
  const panes = el("main");
  const buttons: HTMLButtonElement[] = [];
  const paneEls: HTMLElement[] = [];
  screens.forEach((s, i) => {
    const btn = el("button", { type: "button", role: "tab", "aria-controls": `pane-${s.id}`, "aria-selected": i === 0 ? "true" : "false" }, [s.title]);
    const pane = el("section", { id: `pane-${s.id}`, role: "tabpanel" });
    pane.hidden = i !== 0;
    btn.addEventListener("click", () => {
      buttons.forEach((b, j) => b.setAttribute("aria-selected", j === i ? "true" : "false"));
      paneEls.forEach((p, j) => { p.hidden = j !== i; });
      if (pane.childElementCount === 0) s.draw(doc, pane);
    });
    buttons.push(btn);
    paneEls.push(pane);
    nav.append(btn);
    panes.append(pane);
  });
  root.append(header, nav, panes);
  screens[0]!.draw(doc, paneEls[0]!);
  const dl = el("a", { class: "muted small", href: "#", download: `${doc.provenance.dataset}-report.json` }, ["download the JSON behind this report"]);
  dl.addEventListener("click", (e) => {
    e.preventDefault();
    const blob = new Blob([JSON.stringify(doc)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: dl.getAttribute("download") ?? "report.json" });
    document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  });
  root.append(el("footer", {}, [dl]));
}

const GlasshouseReport: GlasshouseReportApi = { render: renderReport, parse: parseReport, version: "0.0.1" };
(globalThis as unknown as { GlasshouseReport: GlasshouseReportApi }).GlasshouseReport = GlasshouseReport;

// Render the Python-produced fixture through the built dist/report.js in a DOM. If Python
// changes the document's shape, this fails here before any user sees a broken report.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const ROOT = resolve(__dirname, "..");
const FIXTURE = resolve(ROOT, "..", "tests", "fixtures", "report_small.json");
const BINARY_FIXTURE = resolve(ROOT, "..", "tests", "fixtures", "report_binary_small.json");
const DIST = resolve(ROOT, "dist", "report.js");

interface Api {
  render(doc: unknown, root: HTMLElement): void;
  parse(text: string): unknown;
  version: string;
}

function boot(withPlotly: boolean): { dom: JSDOM; api: Api; root: HTMLElement } {
  const dom = new JSDOM(`<!doctype html><html><body><div id="report"></div></body></html>`, { runScripts: "outside-only" });
  const win = dom.window as unknown as Record<string, unknown>;
  if (withPlotly) {
    const calls: unknown[] = [];
    win.Plotly = { newPlot: (root: HTMLElement, data: unknown[]) => { calls.push(data); root.setAttribute("data-plotly", String(data.length)); }, __calls: calls };
  }
  dom.window.eval(readFileSync(DIST, "utf8"));
  const api = win.GlasshouseReport as Api;
  const root = dom.window.document.getElementById("report") as HTMLElement;
  return { dom, api, root };
}

const text = readFileSync(FIXTURE, "utf8");
const binaryText = readFileSync(BINARY_FIXTURE, "utf8");

describe("glasshouse report viewer", () => {
  it("parses the fixture and refuses other schemas", () => {
    const { api } = boot(false);
    const doc = api.parse(text) as { models: string[] };
    expect(doc.models).toEqual(["glm", "mean"]);
    expect(() => api.parse(JSON.stringify({ schema: "glasshouse-report/0", models: ["x"] }))).toThrow(/glasshouse-report\/1/);
    expect(() => api.parse(JSON.stringify({ schema: "glasshouse-report/1", models: [] }))).toThrow(/no models/);
  });

  it("renders the overview with every metric, a naive column and a provenance block", () => {
    const { api, root } = boot(true);
    api.render(api.parse(text), root);
    const heads = Array.from(root.querySelectorAll("table.panel thead th")).map((n) => n.textContent);
    expect(heads).toEqual(["metric", "glm", "mean", "naive"]);
    const metrics = Array.from(root.querySelectorAll("table.panel tbody tr > th")).map((n) => n.textContent?.replace(" ★", ""));
    expect(metrics).toContain("deviance");
    expect(metrics).toContain("gini");
    expect(root.querySelector("tr.primary th")?.textContent).toContain("deviance");
    expect(root.querySelector(".provenance pre")?.textContent).toContain("made up");
    expect(root.querySelectorAll("nav.tabs button").length).toBe(5);
    // the tournament: one row per model, shares that add to 100 %
    const rows = Array.from(root.querySelectorAll("table.tournament tbody tr"));
    expect(rows.map((r) => r.querySelector("th")?.textContent)).toEqual(["glm", "mean"]);
    const shares = rows.map((r) => parseFloat(r.querySelectorAll("td")[0]!.textContent ?? "0"));
    expect(shares.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 0);
  });

  it("model screen shows importance, a partial dependence per feature and the GLM coefficients", () => {
    const { api, root } = boot(true);
    api.render(api.parse(text), root);
    const tab = Array.from(root.querySelectorAll("nav.tabs button")).find((b) => b.textContent === "Model") as HTMLButtonElement;
    tab.click();
    const pane = root.querySelector("#pane-model") as HTMLElement;
    expect(pane.hidden).toBe(false);
    expect(pane.querySelectorAll("[data-plotly]").length).toBe(2);
    const options = Array.from(pane.querySelectorAll("select option")).map((o) => (o as HTMLOptionElement).value);
    expect(options).toEqual(["region", "age"]);
    const tables = pane.querySelectorAll("table.coefficients");
    expect(tables.length).toBe(2);
    expect(tables[0]!.querySelector("tbody th")?.textContent).toBe("intercept");
    expect(Array.from(tables[0]!.querySelectorAll("thead th")).map((n) => n.textContent)).toContain("relativity");
  });

  it("binary reports have no tournament (a probability is not a price)", () => {
    const { api, root } = boot(true);
    api.render(api.parse(binaryText), root);
    expect(root.querySelector("table.tournament")).toBeNull();
  });

  it("compare screen draws the pair's table and two charts", () => {
    const { api, root } = boot(true);
    api.render(api.parse(text), root);
    (root.querySelectorAll("nav.tabs button")[1] as HTMLButtonElement).click();
    const pane = root.querySelector("#pane-compare") as HTMLElement;
    expect(pane.hidden).toBe(false);
    expect(pane.querySelectorAll("table.panel tbody tr").length).toBeGreaterThan(5);
    expect(pane.querySelectorAll("table.tournament tbody tr").length).toBe(2);
    expect(pane.querySelectorAll("[data-plotly]").length).toBe(2);
  });

  it("curves screen lists every curve kind and the A/E-by-feature slices", () => {
    const { api, root } = boot(true);
    api.render(api.parse(text), root);
    (root.querySelectorAll("nav.tabs button")[2] as HTMLButtonElement).click();
    const pane = root.querySelector("#pane-curves") as HTMLElement;
    const options = Array.from(pane.querySelectorAll("select option")).map((o) => (o as HTMLOptionElement).value);
    expect(options).toEqual(
      expect.arrayContaining(["lorenz", "lift", "calibration", "oneway:region", "ae:region", "ae:age", "ae:time"]),
    );
    expect(pane.querySelector("[data-plotly]")).not.toBeNull();
    expect(pane.querySelectorAll(".toggles input").length).toBe(2);
  });

  it("residuals tab shows the summary and two charts per model", () => {
    const { api, root } = boot(true);
    api.render(api.parse(text), root);
    (Array.from(root.querySelectorAll("nav.tabs button")).find((b) => b.textContent === "Residuals") as HTMLButtonElement).click();
    const pane = root.querySelector("#pane-residuals") as HTMLElement;
    expect(pane.hidden).toBe(false);
    const rows = Array.from(pane.querySelectorAll("table.panel tbody tr > th")).map((n) => n.textContent);
    expect(rows).toEqual(["deviance", "pearson"]);
    expect(pane.querySelectorAll("[data-plotly]").length).toBe(2);
    expect(pane.textContent).toContain("Residual vs fitted");
  });

  it("binary reports get a Threshold tab whose slider walks the precomputed grid", () => {
    const { api, root } = boot(true);
    api.render(api.parse(binaryText), root);
    const tabs = Array.from(root.querySelectorAll("nav.tabs button")).map((b) => b.textContent);
    expect(tabs).toContain("Threshold");
    (root.querySelectorAll("nav.tabs button")[4] as HTMLButtonElement).click();
    const pane = root.querySelector("#pane-threshold") as HTMLElement;
    expect(pane.hidden).toBe(false);
    expect(pane.textContent).toContain("alerts per catch");
    const slider = pane.querySelector("input[type=range]") as HTMLInputElement;
    const before = pane.querySelector("table.panel tbody tr td")?.textContent;
    slider.value = "90";
    slider.dispatchEvent(new (pane.ownerDocument.defaultView as any).Event("input"));
    const after = pane.querySelector("table.panel tbody tr td")?.textContent;
    expect(after).not.toBe(before); // flagged count moves with the threshold
    expect(pane.querySelector("strong")?.textContent).toBe("0.9");
  });

  it("non-binary reports have no Threshold tab", () => {
    const { api, root } = boot(false);
    api.render(api.parse(text), root);
    const tabs = Array.from(root.querySelectorAll("nav.tabs button")).map((b) => b.textContent);
    expect(tabs).not.toContain("Threshold");
  });

  it("falls back to tables when Plotly is not available", () => {
    const { api, root } = boot(false);
    api.render(api.parse(text), root);
    (root.querySelectorAll("nav.tabs button")[2] as HTMLButtonElement).click();
    const pane = root.querySelector("#pane-curves") as HTMLElement;
    expect(pane.querySelector("[data-plotly]")).toBeNull();
    expect(pane.querySelector("table.grid")).not.toBeNull();
    expect(pane.textContent).toContain("unavailable offline");
  });
});

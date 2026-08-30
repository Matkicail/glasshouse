"use strict";
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

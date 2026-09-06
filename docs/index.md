# glasshouse

Interpretable, well-rounded ML with a Rust core and a Python API. Glass-box models (GLMs
first), metrics that tell the truth, and one report that always shows whether a model beats
the naive baseline.

```bash
pip install "glasshouse[data]"
```

Three pages, in the order a new reader wants them:

- **[Comparing models](comparing-models.md)**: the report end to end. Two ways in (your own
  predictions, or glasshouse fitting on folds), one worked example per task type, and how
  to read every tab. Every code block on that page runs as a test.
- **[Methods and formulas](methods.md)**: what each number is, with references, and the one
  paragraph on weights that everything else links to.
- **[The report suite](report-suite.md)**: the design of the report, what it shows to whom,
  and what has landed.
- **[The research track](research.md)**: the fenced neural models, starting with the CANN,
  and the rule each one has to pass to leave the fence.

The README on the repository shows the ten-line version and the benchmark tables; the
`COMMANDS.md` file there is the recipe for working on the code.

## The idea in one table

Every score is weighted, every table has the naive row, and the task is declared rather than
guessed, so the report knows which panel to show:

| task | family | naive baseline | the first thing to read |
|---|---|---|---|
| frequency, severity, pure premium | poisson, gamma, tweedie | the weighted mean | double lift against the incumbent, then A/E by segment |
| binary | binomial | the class prior | precision-recall and the threshold's cost in alerts per catch |
| regression | gaussian | the mean | MAE and RMSE against naive, residuals over time |

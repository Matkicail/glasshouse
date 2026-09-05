// The shape of a glasshouse-report/1 document. Hand-written to mirror report/schema.json;
// the vitest suite renders the Python-produced fixture, so a drift between the two fails a test.
// Nothing here is computed: the browser only draws what Python wrote.

type TaskType = "frequency" | "severity" | "pure_premium" | "binary" | "regression";

interface TaskInfo {
  type: TaskType;
  family: string;
  power: number | null;
  threshold: number | null;
  primary_metric: string;
  n_bins: number;
}

interface Provenance {
  dataset: string;
  describe: string;
  split: Record<string, unknown> | null;
  n_rows: number;
  weight_sum: number;
  sample_rows: number;
  sample_seed: number;
}

interface Binned {
  n_rows: number[];
  weight: number[];
  predicted: number[];
  actual: number[];
  actual_over_expected: number[];
}

interface ScorecardDoc {
  label: string;
  family: string;
  n_rows: number;
  weight_sum: number;
  naive_prediction: number;
  metrics: Record<string, number>;
  naive: Record<string, number>;
  calibration: Binned;
}

type CompareRow = [string, number, number, string];

interface Comparison {
  a: string;
  b: string;
  rows: CompareRow[];
}

interface LorenzCurve { kind: "lorenz"; label: string; x: number[]; y: number[]; gini: number }
interface LiftCurve { kind: "lift"; label: string; bin: number[]; weight: number[]; predicted: number[]; actual: number[] }
interface DoubleLiftCurve {
  kind: "double_lift"; label_a: string; label_b: string; bin: number[]; weight: number[];
  ratio: number[]; actual: number[]; predicted_a: number[]; predicted_b: number[];
}
interface CalibrationCurve {
  kind: "calibration"; label: string; bin: number[]; weight: number[]; predicted: number[];
  actual: number[]; actual_over_expected: number[];
}
interface RocCurve { kind: "roc"; label: string; fpr: number[]; tpr: number[]; threshold: number[]; auc: number }
interface PrCurve {
  kind: "pr"; label: string; recall: number[]; precision: number[]; threshold: number[];
  average_precision: number; positive_rate: number;
}
type Curve = LorenzCurve | LiftCurve | DoubleLiftCurve | CalibrationCurve | RocCurve | PrCurve;

interface AEByFeature extends Binned {
  kind: "ae_by_feature";
  feature: string;
  label: string;
  level: string[];
}

interface ResidualStats { mean: number; std: number; median: number; [q: string]: number }

interface ResidualDoc {
  definition: Record<string, string>;
  summary: { deviance: ResidualStats; pearson: ResidualStats };
  histogram: { edges: number[]; counts: number[] };
  scatter: { fitted: number[]; deviance: number[]; actual: number[] };
  by_feature: AEByFeature[];
  over_time: AEByFeature | null;
}

interface BenchBlock {
  summary: Record<string, Record<string, { mean: number; std: number }>>;
  naive_summary?: Record<string, { mean: number; std: number }>;
  folds: { label: string; fold: number; seconds: number }[];
}

interface ThresholdGrid {
  threshold: number[];
  tp: number[];
  fp: number[];
  fn: number[];
  tn: number[];
  accuracy: number[];
  precision: number[];
  recall: number[];
  f1: number[];
  mcc: number[];
  alerts: number[];
  alerts_per_tp: (number | null)[];
}

interface TournamentTable {
  labels: string[];
  share: number[];
  weight: number[];
  predicted: number[];
  actual: number[];
  profit: number[];
  actual_over_expected: (number | null)[];
}

interface TournamentBlock {
  pairs: TournamentTable[];
  overall: TournamentTable;
}

interface PartialDependenceDoc {
  feature: string;
  kind: "numeric" | "categorical";
  grid: (number | string)[];
  mean: number[];
  low: number[];
  high: number[];
}

interface ImportanceDoc {
  features: string[];
  mean: number[];
  std: number[];
  base_deviance: number;
}

interface CoefficientsDoc {
  terms: string[];
  mean: number[];
  std: number[];
  relativity: number[] | null;
}

interface ExplainDoc {
  partial_dependence: PartialDependenceDoc[];
  importance: ImportanceDoc;
  coefficients: CoefficientsDoc | null;
}

interface HistogramDoc {
  name: string;
  level: string[]; // bin labels; the last is the pooled tail when there is one
  edges: number[];
  n_rows: number[];
  weight: number[];
  summary: Record<string, number>; // mean, std, min, q05, q25, median, q75, q95, max, zero_share
}

interface FeatureProfileDoc {
  feature: string;
  kind: "numeric" | "categorical";
  level: string[];
  edges: number[] | null;
  n_rows: number[];
  weight: number[];
  actual: (number | null)[];
  n_levels: number;
}

interface DataBlock {
  target: HistogramDoc;
  weight: HistogramDoc | null;
  features: FeatureProfileDoc[];
}

interface ReportDoc {
  schema: "glasshouse-report/1";
  task: TaskInfo;
  provenance: Provenance;
  models: string[];
  scorecards: Record<string, ScorecardDoc>;
  naive: Record<string, number>;
  comparisons: Comparison[];
  curves: Curve[];
  residuals: Record<string, ResidualDoc>;
  bench?: BenchBlock;
  thresholds?: Record<string, ThresholdGrid>;
  tournament?: TournamentBlock;
  explain?: Record<string, ExplainDoc>;
  data?: DataBlock;
}

// Direction of "better" per metric; mirrors glasshouse.scorecard.HIGHER_IS_BETTER.
const HIGHER_IS_BETTER: Record<string, boolean> = {
  deviance: false, d2: true, gini: true, normalized_gini: true, rmse: false, mae: false, r2: true,
  mcc: true, f1: true, roc_auc: true, average_precision: true, ks: true, log_loss: false, brier: false,
};

//! The row passes every solver shares: parallel over fixed-size chunks (rayon), partial
//! sums combined in chunk order, so a result is bit-for-bit the same whatever the thread
//! count. Reproducible first, fast second.

use rayon::prelude::*;

/// Rows per parallel chunk. Small enough to feed every core on a modest fold, large enough
/// that the per-chunk `p x p` partials are noise next to the row work.
pub const CHUNK: usize = 4096;

/// The row ranges `[lo, hi)` of the fixed-size chunks, in order.
#[must_use]
pub fn chunks(n_rows: usize) -> impl IndexedParallelIterator<Item = (usize, usize)> {
    (0..n_rows.div_ceil(CHUNK))
        .into_par_iter()
        .map(move |c| (c * CHUNK, ((c + 1) * CHUNK).min(n_rows)))
}

/// One value per row, computed in parallel; the order of the output is the order of the rows.
pub fn per_row(n_rows: usize, f: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
    let mut out = vec![0.0; n_rows];
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, chunk)| {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = f(c * CHUNK + k);
            }
        });
    out
}

/// `sum_i f(i)`: sequential within each chunk, chunk partials added in chunk order.
pub fn chunk_sum(n_rows: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partials: Vec<f64> = chunks(n_rows)
        .map(|(lo, hi)| (lo..hi).map(&f).sum::<f64>())
        .collect();
    partials.iter().sum()
}

/// `target[i] -= factor * col[i]` for every row, in parallel; element-wise, so the thread
/// count cannot change a bit.
pub fn axpy(target: &mut [f64], col: &[f64], factor: f64) {
    target
        .par_chunks_mut(CHUNK)
        .zip(col.par_chunks(CHUNK))
        .for_each(|(t, c)| {
            for (ti, ci) in t.iter_mut().zip(c) {
                *ti += factor * ci;
            }
        });
}

/// The design transposed: row-major `n_rows x p` in, row-major `p x n_rows` out, so each
/// column becomes contiguous. Parallel over columns.
#[must_use]
pub fn transpose(x: &[f64], n_rows: usize, p: usize) -> Vec<f64> {
    let cols: Vec<Vec<f64>> = (0..p)
        .into_par_iter()
        .map(|j| (0..n_rows).map(|i| x[i * p + j]).collect())
        .collect();
    cols.concat()
}

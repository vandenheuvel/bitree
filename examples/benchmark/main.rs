//! Side-by-side micro-benchmark across four Fenwick/BIT crates.
//!
//! Run with:
//!
//! ```text
//! cd examples/benchmark
//! cargo run --release
//! ```
//!
//! Fairness notes:
//!
//! * All 16 (crate × phase) measurements are pooled and run in a fresh
//!   pseudo-random order every repetition, so no crate is systematically
//!   measured "first" or "last" in the cycle.
//! * A few warm-up repetitions run before measurement begins, to let the
//!   CPU frequency, branch predictors, and caches reach steady state.
//! * The `build` phase excludes the source-`Vec` clone (or, for `fenwick`,
//!   the zero-initialised backing buffer) from the timed region — that's
//!   O(n) memcpy/memset that would otherwise dominate.
//! * The `add_at` phase likewise excludes the tree-build setup.
//! * The `restore` phase excludes the tree-build setup for the crate whose
//!   primitive consumes the tree (`bitree`); the others operate on a
//!   shared pre-built tree.
//! * The reported number is the best wall-clock time across `REPS`
//!   measured repetitions, which further damps jitter.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use bitree::BITree;
use fenwick::array::{prefix_sum as fen_prefix_sum, update as fen_update};
use ftree::FenwickTree;
use segment_tree::PrefixPoint;
use segment_tree::ops::Add;

const N: usize = 1 << 20;
const OPS: usize = 1 << 20;
const WARMUP: usize = 11;
const REPS: usize = 401;
/// Reported value per (crate, phase) is the average of the `BEST_N`
/// fastest of the `REPS` samples — robust to high-outlier jitter while
/// smoothing out the single-minimum bias of "best of N".
const BEST_N: usize = 41;

const CRATES: [&str; 4] = ["bitree", "ftree", "segment-tree", "fenwick"];
const PHASES: [(&str, usize); 4] = [
    ("build", N),
    ("prefix_sum", OPS),
    ("add_at", OPS),
    ("restore", N),
];

type Key = (&'static str, &'static str);
type Samples = BTreeMap<Key, Vec<Duration>>;

fn values(n: usize) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(s >> 40);
    }
    v
}

fn indices(n: usize, bound: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(n);
    let mut s: u64 = 0xDEADBEEFCAFEBABE;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push((s as usize) % bound);
    }
    v
}

/// Fisher–Yates shuffle with a deterministic splitmix-style PRNG so each
/// rep gets a different — but reproducible — task order.
fn shuffle(tasks: &mut [(usize, usize)], seed: u64) {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    for i in (1..tasks.len()).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((s >> 32) as usize) % (i + 1);
        tasks.swap(i, j);
    }
}

fn record(samples: &mut Samples, crate_name: &'static str, phase: &'static str, d: Duration) {
    samples.entry((crate_name, phase)).or_default().push(d);
}

/// Average of the fastest `n` samples from `durations`.
fn avg_of_best(durations: &[Duration], n: usize) -> Duration {
    let mut sorted: Vec<Duration> = durations.to_vec();
    sorted.sort();
    let take = n.min(sorted.len());
    let sum: Duration = sorted.iter().take(take).sum();
    sum / take as u32
}

fn ns_per_op(d: Duration, ops: usize) -> f64 {
    d.as_secs_f64() * 1e9 / ops as f64
}

struct Ctx<'a> {
    input: &'a [u64],
    u_idx: &'a [usize],
    u_val: &'a [u64],
    q_idx_excl: &'a [usize],
    q_idx_incl: &'a [usize],
    bi_q: &'a BITree<u64>,
    ft_q: &'a FenwickTree<u64>,
    sg_q: &'a PrefixPoint<u64, Add>,
    fn_q: &'a [u64],
}

/// Run a single (phase, crate) measurement and return its wall-clock.
/// Any O(n) setup that doesn't belong to the operation under test happens
/// *before* `Instant::now()`.
fn measure(ctx: &Ctx, phase: usize, cr: usize) -> Duration {
    match (phase, cr) {
        // ---------- build ----------
        (0, 0) => {
            let buf = ctx.input.to_vec();
            let t = Instant::now();
            black_box(BITree::<u64>::from(buf));
            t.elapsed()
        }
        (0, 1) => {
            let buf = ctx.input.to_vec();
            let t = Instant::now();
            black_box(FenwickTree::<u64>::from_iter(buf));
            t.elapsed()
        }
        (0, 2) => {
            let buf = ctx.input.to_vec();
            let t = Instant::now();
            black_box(PrefixPoint::build(buf, Add));
            t.elapsed()
        }
        (0, 3) => {
            let mut buf = vec![0u64; N];
            let t = Instant::now();
            for (i, &v) in ctx.input.iter().enumerate() {
                fen_update(&mut buf, i, v);
            }
            black_box(&buf);
            t.elapsed()
        }

        // ---------- prefix_sum ----------
        (1, 0) => {
            let t = Instant::now();
            let mut acc: u64 = 0;
            for &i in ctx.q_idx_excl {
                acc = acc.wrapping_add(ctx.bi_q.prefix_sum(i));
            }
            black_box(acc);
            t.elapsed()
        }
        (1, 1) => {
            let t = Instant::now();
            let mut acc: u64 = 0;
            for &i in ctx.q_idx_excl {
                acc = acc.wrapping_add(ctx.ft_q.prefix_sum(i, 0));
            }
            black_box(acc);
            t.elapsed()
        }
        (1, 2) => {
            let t = Instant::now();
            let mut acc: u64 = 0;
            for &i in ctx.q_idx_incl {
                acc = acc.wrapping_add(ctx.sg_q.query(i));
            }
            black_box(acc);
            t.elapsed()
        }
        (1, 3) => {
            let t = Instant::now();
            let mut acc: u64 = 0;
            for &i in ctx.q_idx_incl {
                acc = acc.wrapping_add(fen_prefix_sum(ctx.fn_q, i));
            }
            black_box(acc);
            t.elapsed()
        }

        // ---------- add_at ----------
        // The tree is rebuilt per rep (updates are destructive) but the
        // build happens outside the timer.
        (2, 0) => {
            let mut tree = BITree::<u64>::from(ctx.input.to_vec());
            let t = Instant::now();
            for (&i, &v) in ctx.u_idx.iter().zip(ctx.u_val.iter()) {
                tree.add_at(i, v);
            }
            black_box(&tree);
            t.elapsed()
        }
        (2, 1) => {
            let mut tree = FenwickTree::<u64>::from_iter(ctx.input.to_vec());
            let t = Instant::now();
            for (&i, &v) in ctx.u_idx.iter().zip(ctx.u_val.iter()) {
                tree.add_at(i, v);
            }
            black_box(&tree);
            t.elapsed()
        }
        (2, 2) => {
            let mut tree = PrefixPoint::build(ctx.input.to_vec(), Add);
            let t = Instant::now();
            for (&i, &v) in ctx.u_idx.iter().zip(ctx.u_val.iter()) {
                tree.modify(i, v);
            }
            black_box(&tree);
            t.elapsed()
        }
        (2, 3) => {
            let mut fw = vec![0u64; N];
            for (i, &v) in ctx.input.iter().enumerate() {
                fen_update(&mut fw, i, v);
            }
            let t = Instant::now();
            for (&i, &v) in ctx.u_idx.iter().zip(ctx.u_val.iter()) {
                fen_update(&mut fw, i, v);
            }
            black_box(&fw);
            t.elapsed()
        }

        // ---------- restore ----------
        (3, 0) => {
            // `Vec::from(tree)` consumes; rebuild fresh outside the timer.
            let tree = BITree::<u64>::from(ctx.input.to_vec());
            let t = Instant::now();
            black_box(Vec::<u64>::from(tree));
            t.elapsed()
        }
        (3, 1) => {
            let t = Instant::now();
            let mut out = Vec::<u64>::with_capacity(N);
            let mut prev = 0u64;
            for i in 0..N {
                let cur = ctx.ft_q.prefix_sum(i + 1, 0);
                out.push(cur - prev);
                prev = cur;
            }
            black_box(out);
            t.elapsed()
        }
        (3, 2) => {
            let t = Instant::now();
            let mut out = Vec::<u64>::with_capacity(N);
            let mut prev = 0u64;
            for i in 0..N {
                let cur = ctx.sg_q.query(i);
                out.push(cur - prev);
                prev = cur;
            }
            black_box(out);
            t.elapsed()
        }
        (3, 3) => {
            let t = Instant::now();
            let mut out = Vec::<u64>::with_capacity(N);
            let mut prev = 0u64;
            for i in 0..N {
                let cur: u64 = fen_prefix_sum(ctx.fn_q, i);
                out.push(cur - prev);
                prev = cur;
            }
            black_box(out);
            t.elapsed()
        }

        _ => unreachable!(),
    }
}

fn main() {
    let input = values(N);
    let q_idx_excl = indices(OPS, N + 1);
    let q_idx_incl = indices(OPS, N);
    let u_idx = indices(OPS, N);
    let u_val: Vec<u64> = input.iter().take(OPS).copied().collect();

    // Pre-built structures for the read-only `prefix_sum` and for the
    // three `restore` paths that iterate without consuming the tree.
    let bi_q = BITree::<u64>::from(input.clone());
    let ft_q = FenwickTree::<u64>::from_iter(input.clone());
    let sg_q = PrefixPoint::build(input.clone(), Add);
    let fn_q = {
        let mut fw = vec![0u64; N];
        for (i, &v) in input.iter().enumerate() {
            fen_update(&mut fw, i, v);
        }
        fw
    };

    let ctx = Ctx {
        input: &input,
        u_idx: &u_idx,
        u_val: &u_val,
        q_idx_excl: &q_idx_excl,
        q_idx_incl: &q_idx_incl,
        bi_q: &bi_q,
        ft_q: &ft_q,
        sg_q: &sg_q,
        fn_q: &fn_q,
    };

    // Build the full task list: every (phase, crate) pair.
    let mut base_tasks: Vec<(usize, usize)> = Vec::with_capacity(PHASES.len() * CRATES.len());
    for p in 0..PHASES.len() {
        for c in 0..CRATES.len() {
            base_tasks.push((p, c));
        }
    }

    let mut samples: Samples = BTreeMap::new();

    for rep in 0..(WARMUP + REPS) {
        let mut tasks = base_tasks.clone();
        shuffle(&mut tasks, rep as u64);
        for &(p, c) in &tasks {
            let d = measure(&ctx, p, c);
            if rep >= WARMUP {
                record(&mut samples, CRATES[c], PHASES[p].0, d);
            }
        }
    }

    // Collapse the per-(crate, phase) sample vector into a single number.
    let summary: BTreeMap<Key, Duration> = samples
        .iter()
        .map(|(k, v)| (*k, avg_of_best(v, BEST_N)))
        .collect();

    println!(
        "n = {N}, ops = {OPS}, warmup = {WARMUP}, reps = {REPS}, reported = avg of best {BEST_N}\n\
         task order shuffled per rep; O(n) setup for build / add_at / bitree's restore\n\
         excluded from the timed region.\n"
    );

    // Header.
    print!("{:<14}", "crate");
    for (name, _) in &PHASES {
        print!(" {:>14}", name);
    }
    println!();
    println!("{}", "-".repeat(14 + PHASES.len() * 15));

    // Find the fastest value per phase so we can mark the winner.
    let mut winners: BTreeMap<&'static str, &'static str> = BTreeMap::new();
    for (phase, _) in &PHASES {
        let mut best_c: Option<(&'static str, Duration)> = None;
        for &c in &CRATES {
            if let Some(&d) = summary.get(&(c, *phase)) {
                if best_c.map_or(true, |(_, b)| d < b) {
                    best_c = Some((c, d));
                }
            }
        }
        if let Some((c, _)) = best_c {
            winners.insert(phase, c);
        }
    }

    // Body.
    for &c in &CRATES {
        print!("{:<14}", c);
        for &(phase, ops) in &PHASES {
            match summary.get(&(c, phase)) {
                Some(&d) => {
                    let ns = ns_per_op(d, ops);
                    let marker = if winners.get(phase) == Some(&c) { "*" } else { " " };
                    print!(" {:>11.2} ns{}", ns, marker);
                }
                None => print!(" {:>14}", "—"),
            }
        }
        println!();
    }

    println!("\nAll numbers are ns/op.  `*` marks the fastest crate in each column.");
}

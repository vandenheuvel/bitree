# bitree

A small, `no_std`-friendly **Binary Indexed Tree** for Rust — the same data
structure more commonly known as a **Fenwick tree**.

A Fenwick tree maintains prefix sums over a mutable sequence. Both point
updates and prefix queries run in `O(log n)` time, and the whole structure
fits in a single `Vec<T>` the same length as the input array.

- A [**prefix sum array**](https://en.wikipedia.org/wiki/Prefix_sum) answers
  range-sum queries in `O(1)` but costs `O(n)` per update — good for static
  data, bad for streaming.
- A [**segment tree**](https://en.wikipedia.org/wiki/Segment_tree) is a
  strictly more general cousin (arbitrary monoids, range updates), at the
  cost of `~2×` memory and a heavier implementation.

A Fenwick tree sits somewhere in between: `O(log n)` for both queries
and updates, with no memory overhead versus the original array. See [Fenwick tree on Wikipedia](https://en.wikipedia.org/wiki/Fenwick_tree).

## Features

- `no_std` compatible (uses `alloc`).
- Generic over any `T` implementing `AddAssign<&T>` / `SubAssign<&T>` (like `f32/f64`, `usize`, ...).
- No `Copy` needed, so works for e.g., arbitrary precision types.
- Dynamic sizing via `push` / `pop`, just like (the underlying) `Vec`.
- Inverse lookup: given a prefix sum, find the slot that contains it.
- **Efficient consumption in both directions** — see below.
- Optional `serde` support behind the `serde` feature.

## API at a glance

| Operation | Method | Complexity |
| --- | --- | --- |
| Build from values | `BITree::from_iter` | `O(n)` |
| Create empty | `BITree::new` | `O(1)` |
| Prefix sum `[0..i)` | `prefix_sum(i)` | `O(log n)` |
| Accumulate prefix sum into an existing value | `add_prefix_sum` / `sub_prefix_sum` | `O(log n)` |
| Point update | `add_at(i, v)` / `sub_at(i, v)` | `O(log n)` |
| Append / truncate | `push(v)` / `pop()` | amortised `O(log n)` |
| Invert a prefix sum | `binary_search(s)` / `sub_binary_search` | `O(log n)` |
| Consume into an iterator over the original values | `into_iter()` (+ `.rev()`) | `O(n)` total |

## Efficient iteration

`IntoIterator::into_iter` runs an `O(n)` in-place un-build that inverts the Fenwick construction, then hands you a plain
`Vec<T>` iterator. That iterator is **both `DoubleEndedIterator` and `ExactSizeIterator`**, so you get forward and
reverse traversal at `O(1)` per element:

```rust
use bitree::BITree;

let bitree = BITree::from_iter([1, 6, 3, 9, 2]);

let forward: Vec<_> = bitree.clone().into_iter().collect();
assert_eq!(forward, vec![1, 6, 3, 9, 2]);

let reversed: Vec<_> = bitree.into_iter().rev().collect();
assert_eq!(reversed, vec![2, 9, 3, 6, 1]);
```

## Demo

```rust
use bitree::BITree;

// Build from a slice — values are the per-slot counts.
let mut bitree = BITree::from_iter([1usize, 6, 3, 9, 2]);

// Prefix sums in O(log n). prefix_sum(i) covers [0..i).
assert_eq!(bitree.prefix_sum(0), 0);
assert_eq!(bitree.prefix_sum(3), 10); // 1 + 6 + 3
assert_eq!(bitree.prefix_sum(5), 21); // full sum

// Point update: add 5 to slot 1. Logical array becomes [1, 11, 3, 9, 2].
bitree.add_at(1, 5);
assert_eq!(bitree.prefix_sum(3), 15);

// Grow and shrink dynamically.
bitree.push(4);
assert_eq!(bitree.prefix_sum(6), 30);
bitree.pop();
assert_eq!(bitree.prefix_sum(5), 26);

// Inverse lookup: binary-search the prefix sums [0, 1, 7, 10, 19, 21].
// `Ok(k)` means target == prefix_sum(k); `Err(k)` is the insertion point.
let bitree = BITree::from_iter([1usize, 6, 3, 9, 2]);
assert_eq!(bitree.binary_search(7), Ok(2));  // exact boundary
assert_eq!(bitree.binary_search(9), Err(3)); // falls inside slot 2

// Or drive the walk in-place to inspect the remainder yourself.
let mut remaining = 9usize;
let pos = bitree.sub_binary_search(&mut remaining);
assert_eq!((pos, remaining), (2, 2)); // 1 + 6 = 7, then 2 more into slot 2.

// Walk the original values back out, cheaply, in either direction.
let forward: Vec<_> = bitree.clone().into_iter().collect();
assert_eq!(forward, vec![1, 6, 3, 9, 2]);
let reversed: Vec<_> = bitree.into_iter().rev().collect();
assert_eq!(reversed, vec![2, 9, 3, 6, 1]);
```

## Acknowledgements

This crate derives from the [`ftree`](https://crates.io/crates/ftree) crate.

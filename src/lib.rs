//! A binary indexed tree (Fenwick tree) for efficient prefix sums.
//!
//! A [`BITree<T>`] maintains prefix sums over a mutable sequence of values.
//! Both point updates and prefix queries run in *O*(log *n*) time, and the
//! whole structure lives in a single [`Vec<T>`] the same length as the
//! logical array.
//!
//! This crate is `no_std`-compatible; it depends only on [`alloc`]. Values are
//! generic over any `T` implementing `AddAssign<&T>` — and, where a method
//! requires subtraction, `SubAssign<&T>`. `T: Copy` is not required, so
//! arbitrary-precision integers and similar types are supported.
//!
//! # Examples
//!
//! ```
//! use bitree::BITree;
//!
//! let mut bitree = BITree::from_iter([1, 6, 3, 9, 2]);
//!
//! // Prefix sums in O(log n); `prefix_sum(i)` covers `[0..i)`.
//! assert_eq!(bitree.prefix_sum(3), 10);
//! assert_eq!(bitree.prefix_sum(5), 21);
//!
//! // Point update.
//! bitree.add_at(1, 5);
//! assert_eq!(bitree.prefix_sum(3), 15);
//!
//! // Grow and shrink like a `Vec`.
//! bitree.push(4);
//! bitree.pop();
//!
//! // Recover the original values.
//! assert_eq!(Vec::from(bitree), vec![1, 11, 3, 9, 2]);
//! ```
//!
//! # Crate features
//!
//! - **`serde`** — derive [`Serialize`] and [`Deserialize`] for [`BITree<T>`].
//!
//! [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
//! [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::ops::{AddAssign, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A binary indexed tree (Fenwick tree) over a sequence of `T` values.
///
/// Conceptually, a `BITree<T>` represents an array `a[0..n]` while internally
/// storing partial sums that allow both prefix-sum queries and point updates
/// in *O*(log *n*). The internal buffer has the same length as the logical
/// array, so no memory is wasted relative to a plain [`Vec<T>`].
///
/// See the [crate-level documentation](crate) for an overview and examples.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct BITree<T> {
    inner: Vec<T>,
}

impl<T> BITree<T> {
    /// Constructs a new, empty `BITree<T>`.
    ///
    /// The tree will not allocate until values are pushed into it.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree: BITree<i32> = BITree::new();
    /// assert!(bitree.is_empty());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Constructs a new `BITree<T>` of length `n`, filled with `T::default()`.
    ///
    /// The initial capacity is exactly `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::<usize>::new_zeros(5);
    /// assert_eq!(bitree.len(), 5);
    /// assert_eq!(bitree.prefix_sum(5), 0);
    /// ```
    #[inline]
    pub fn new_zeros(n: usize) -> Self
    where
        T: Default,
    {
        let mut inner = Vec::with_capacity(n);
        for _ in 0..n {
            inner.push(T::default());
        }

        Self { inner }
    }

    /// Constructs a new, empty `BITree<T>` with at least the specified capacity.
    ///
    /// The tree will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements
    /// than `capacity`. If `capacity` is zero, the tree will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree: BITree<i32> = BITree::with_capacity(10);
    /// assert!(bitree.is_empty());
    ///
    /// for i in 0..10 {
    ///     bitree.push(i);
    /// }
    /// assert_eq!(bitree.len(), 10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Returns `true` if the tree contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::new();
    /// assert!(bitree.is_empty());
    ///
    /// bitree.push(1);
    /// assert!(!bitree.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 2, 3]);
    /// assert_eq!(bitree.len(), 3);
    /// ```
    #[inline]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Removes the last element from the tree, returning `true` if one was
    /// removed and `false` if the tree was already empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3, 9]);
    ///
    /// assert_eq!(bitree.pop(), true);
    /// assert_eq!(bitree.prefix_sum(3), 10); // sum of remaining [1, 6, 3]
    ///
    /// assert_eq!(bitree.pop(), true);
    /// assert_eq!(bitree.prefix_sum(2), 7);  // sum of remaining [1, 6]
    ///
    /// bitree.pop();
    /// bitree.pop();
    /// assert_eq!(bitree.pop(), false);      // already empty
    /// ```
    #[inline(always)]
    pub fn pop(&mut self) -> bool {
        self.inner.pop().is_some()
    }

    #[inline(always)]
    fn walk_prefix<F: FnMut(&mut T, &T)>(&self, index: usize, sum: &mut T, mut op: F) {
        assert!(index < self.inner.len() + 1);

        let mut current_idx = index;
        while current_idx > 0 {
            op(sum, &self.inner[current_idx - 1]);
            current_idx &= current_idx - 1;
        }
    }

    #[inline(always)]
    fn walk_update<F: FnMut(&mut T, &T)>(&mut self, index: usize, diff: T, mut op: F) {
        assert!(index < self.len());

        let mut current_idx = index;
        while let Some(value) = self.inner.get_mut(current_idx) {
            op(value, &diff);
            current_idx |= current_idx + 1;
        }
    }
}

impl<T> Default for BITree<T> {
    /// Creates an empty `BITree<T>`.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: for<'a> AddAssign<&'a T>> From<Vec<T>> for BITree<T> {
    /// Builds a `BITree<T>` from a [`Vec`] of values, reusing its allocation.
    ///
    /// Runs in *O*(*n*).
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let values = vec![1, 6, 3, 9, 2];
    /// let bitree = BITree::from(values);
    /// assert_eq!(bitree.prefix_sum(4), 19);
    /// ```
    #[inline]
    fn from(mut inner: Vec<T>) -> Self {
        let n = inner.len();
        rebuild(&mut inner, 0..n, |p, c| *p += c);
        BITree { inner }
    }
}

impl<T: for<'a> SubAssign<&'a T>> From<BITree<T>> for Vec<T> {
    /// Recovers the original values as a [`Vec`], reusing the tree's allocation.
    ///
    /// Runs in *O*(*n*).
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from(vec![1, 6, 3, 9, 2]);
    /// assert_eq!(Vec::from(bitree), vec![1, 6, 3, 9, 2]);
    /// ```
    #[inline]
    fn from(mut bitree: BITree<T>) -> Self {
        let n = bitree.inner.len();
        rebuild(&mut bitree.inner, (0..n).rev(), |p, c| *p -= c);
        bitree.inner
    }
}

impl<T: for<'a> AddAssign<&'a T>> FromIterator<T> for BITree<T> {
    /// Builds a `BITree<T>` from the values yielded by an iterator.
    ///
    /// Runs in *O*(*n*).
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// assert_eq!(bitree.prefix_sum(5), 21);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<T: for<'a> SubAssign<&'a T>> IntoIterator for BITree<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    /// Consumes the tree and yields the original values in order.
    ///
    /// The returned iterator implements both [`DoubleEndedIterator`] and
    /// [`ExactSizeIterator`], so forward and reverse traversal both run in
    /// *O*(1) per element after an *O*(*n*) setup that inverts the Fenwick
    /// build.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// let collected: Vec<_> = bitree.into_iter().collect();
    /// assert_eq!(collected, vec![1, 6, 3, 9, 2]);
    /// ```
    ///
    /// Reverse iteration works too:
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// let reversed: Vec<_> = bitree.into_iter().rev().collect();
    /// assert_eq!(reversed, vec![2, 9, 3, 6, 1]);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Vec::from(self).into_iter()
    }
}

#[inline(always)]
fn rebuild<T, I, F>(inner: &mut [T], indices: I, mut op: F)
where
    I: IntoIterator<Item = usize>,
    F: FnMut(&mut T, &T),
{
    let n = inner.len();
    let ptr = inner.as_mut_ptr();
    for i in indices {
        let parent = i | (i + 1);
        if parent < n {
            // SAFETY:
            //  - i < parent < n, so both offsets are in-bounds of `inner`.
            //  - parent != i, so the &mut and & never alias.
            //  - `ptr` is derived from a valid &mut [T] that outlives the loop.
            unsafe {
                let child = &*ptr.add(i);
                let parent_ref = &mut *ptr.add(parent);
                op(parent_ref, child);
            }
        }
    }
}

impl<T: for<'a> AddAssign<&'a T> + for<'a> SubAssign<&'a T>> BITree<T> {
    /// Adds the prefix sum of `[0..index)` into `sum`.
    ///
    /// When `index` is `0`, `sum` is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    ///
    /// let mut running = 100;
    /// bitree.add_prefix_sum(3, &mut running);
    /// assert_eq!(running, 110);
    /// ```
    #[inline]
    pub fn add_prefix_sum(&self, index: usize, sum: &mut T) {
        self.walk_prefix(index, sum, |s, v| *s += v);
    }

    /// Subtracts the prefix sum of `[0..index)` from `sum`.
    ///
    /// When `index` is `0`, `sum` is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    ///
    /// let mut running = 100;
    /// bitree.sub_prefix_sum(3, &mut running);
    /// assert_eq!(running, 90);
    /// ```
    #[inline]
    pub fn sub_prefix_sum(&self, index: usize, sum: &mut T) {
        self.walk_prefix(index, sum, |s, v| *s -= v);
    }

    /// Returns the prefix sum of `[0..index)`.
    ///
    /// Equivalent to starting from `T::default()` and calling
    /// [`add_prefix_sum`](Self::add_prefix_sum). The prefix sum at `index = 0`
    /// is `T::default()`.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    ///
    /// assert_eq!(bitree.prefix_sum(0), 0);
    /// assert_eq!(bitree.prefix_sum(1), 1);
    /// assert_eq!(bitree.prefix_sum(3), 10);
    /// assert_eq!(bitree.prefix_sum(5), 21);
    /// ```
    #[inline]
    pub fn prefix_sum(&self, index: usize) -> T
    where
        T: Default,
    {
        let mut sum = T::default();
        self.add_prefix_sum(index, &mut sum);
        sum
    }

    /// Adds `diff` to the value at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// // logical array: [1, 6, 3, 9, 2]
    ///
    /// bitree.add_at(1, 5);
    /// // logical array: [1, 11, 3, 9, 2]
    ///
    /// assert_eq!(bitree.prefix_sum(3), 15);
    /// ```
    #[inline]
    pub fn add_at(&mut self, index: usize, diff: T) {
        self.walk_update(index, diff, |v, d| *v += d);
    }

    /// Subtracts `diff` from the value at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// // logical array: [1, 6, 3, 9, 2]
    ///
    /// bitree.sub_at(3, 4);
    /// // logical array: [1, 6, 3, 5, 2]
    ///
    /// assert_eq!(bitree.prefix_sum(5), 17);
    /// ```
    #[inline]
    pub fn sub_at(&mut self, index: usize, diff: T) {
        self.walk_update(index, diff, |v, d| *v -= d);
    }

    /// Appends a value to the end of the tree.
    ///
    /// Runs in amortised *O*(log *n*).
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3]);
    /// bitree.push(9);
    /// bitree.push(2);
    ///
    /// assert_eq!(bitree.prefix_sum(4), 19);
    /// assert_eq!(bitree.prefix_sum(5), 21);
    /// ```
    #[inline]
    pub fn push(&mut self, mut value: T) {
        let n = self.inner.len();
        for i in 0..n.trailing_ones() {
            let child = n & !(1 << i);
            value += &self.inner[child];
        }
        self.inner.push(value);
    }
}

impl<T: for<'a> AddAssign<&'a T> + for<'a> SubAssign<&'a T> + PartialOrd> BITree<T> {
    /// Walks the tree to find the largest `pos` such that
    /// `prefix_sum(pos) <= target`, subtracting the consumed segment sums
    /// from `*remainder` along the way.
    ///
    /// The caller supplies the initial `target` in `*remainder`. After the
    /// call, `*remainder` holds `target - prefix_sum(pos)`. A remainder of
    /// `T::default()` means `target` landed exactly on the boundary
    /// `prefix_sum(pos)`; any positive remainder falls strictly inside the
    /// slot that starts at `pos`.
    ///
    /// This is the lower-level primitive behind
    /// [`binary_search`](Self::binary_search); use that method when you only
    /// need the slot index and don't care about the remainder.
    ///
    /// Equality is decided via the trichotomy of [`PartialOrd`], so
    /// [`PartialEq`] is not required. Values that are incomparable with
    /// `target` (for example `f64::NAN`) are the caller's responsibility.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    ///
    /// // 9 lies between prefix_sum(2) = 7 and prefix_sum(3) = 10.
    /// let mut remaining = 9;
    /// let idx = bitree.sub_binary_search(&mut remaining);
    /// assert_eq!((idx, remaining), (2, 2));
    ///
    /// // Exact boundary: 7 == prefix_sum(2), so the walk advances past it.
    /// let mut remaining = 7;
    /// let idx = bitree.sub_binary_search(&mut remaining);
    /// assert_eq!((idx, remaining), (2, 0));
    /// ```
    pub fn sub_binary_search(&self, remainder: &mut T) -> usize {
        let n = self.inner.len();
        let mut pos = 0;

        let mut mask = n.checked_ilog2().map_or(0, |log| 1 << log);

        while mask > 0 {
            let next = pos + mask;
            if next <= n {
                let value = &self.inner[next - 1];
                if !(*remainder < *value) {
                    pos = next;
                    *remainder -= value;
                }
            }
            mask >>= 1;
        }

        pos
    }

    /// Binary-searches the conceptual prefix-sum slice
    /// `[prefix_sum(0), prefix_sum(1), ..., prefix_sum(n)]` for `target`.
    ///
    /// The behaviour mirrors [`slice::binary_search`]:
    ///
    /// - `Ok(k)` if `prefix_sum(k) == target`.
    /// - `Err(k)` if `target` would be inserted at position `k` to keep the
    ///   slice sorted — that is, `prefix_sum(k - 1) < target < prefix_sum(k)`,
    ///   with `Err(0)` when `target` is below `prefix_sum(0)` and
    ///   `Err(n + 1)` when `target` is above `prefix_sum(n)`.
    ///
    /// Equality is decided via the trichotomy of [`PartialOrd`], so
    /// [`PartialEq`] is not required. Values that are incomparable with
    /// `target` (for example `f64::NAN`) are the caller's responsibility.
    ///
    /// # Examples
    ///
    /// Looking up sums in an integer tree:
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    /// // prefix sums: 0, 1, 7, 10, 19, 21
    ///
    /// assert_eq!(bitree.binary_search(0), Ok(0));
    /// assert_eq!(bitree.binary_search(7), Ok(2));
    /// assert_eq!(bitree.binary_search(21), Ok(5));
    ///
    /// assert_eq!(bitree.binary_search(6), Err(2));
    /// assert_eq!(bitree.binary_search(9), Err(3));
    /// assert_eq!(bitree.binary_search(22), Err(6));
    /// ```
    ///
    /// An empty tree still has the single prefix sum `prefix_sum(0) = 0`:
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let empty = BITree::<f64>::new();
    ///
    /// assert!(empty.is_empty());
    /// assert_eq!(empty.binary_search(-1.0), Err(0));
    /// assert_eq!(empty.binary_search(0.0), Ok(0));
    /// assert_eq!(empty.binary_search(1.0), Err(1));
    /// ```
    #[inline]
    pub fn binary_search(&self, mut target: T) -> Result<usize, usize>
    where
        T: Default,
    {
        let pos = self.sub_binary_search(&mut target);
        let zero = T::default();
        if target < zero {
            Err(pos)
        } else if zero < target {
            Err(pos + 1)
        } else {
            Ok(pos)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::BITree;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_new() {
        let lengths: [usize; 5] = [1, 6, 3, 9, 2];
        let expected_index = vec![1, 7, 3, 19, 2];
        let actual_index = BITree::from_iter(lengths);
        assert_eq!(expected_index, actual_index.inner);

        let n = 5;
        let tree = BITree::<usize>::new_zeros(5);
        assert_eq!(tree.len(), n);
        assert!(!tree.is_empty());
        assert_eq!(tree.prefix_sum(0), 0);
        assert_eq!(tree.prefix_sum(3), 0);
        assert_eq!(tree.prefix_sum(5), 0);
    }

    #[test]
    fn test_prefix_sum() {
        let lengths = [1, 6, 3, 9, 2];
        let bitree = BITree::from_iter(lengths);

        let cases: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 7), (3, 10), (4, 19), (5, 21)];
        // The prefix sum up until the zeroth element is 0, since there is nothing before it
        // The prefix sum up until an index larger than the length is undefined, since every
        // element after the length - 1 is undefined
        cases
            .into_iter()
            .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum))
    }

    #[test]
    fn test_update_index() {
        let lengths = [1, 6, 3, 9, 2];
        let mut bitree = BITree::from_iter(lengths);

        let cases: Vec<(usize, usize)> = vec![(0, 2), (1, 8), (2, 3), (3, 20), (4, 2)];

        bitree.add_at(0, 1);

        cases
            .into_iter()
            .for_each(|(idx, expected_value)| assert_eq!(bitree.inner[idx], expected_value))
    }

    #[test]
    fn test_binary_search() {
        let lengths = [1, 6, 3, 9, 2];
        let bitree = BITree::from_iter(lengths);
        // prefix sums: 0, 1, 7, 10, 19, 21

        let cases: Vec<(usize, Result<usize, usize>)> = vec![
            (0, Ok(0)),
            (1, Ok(1)),
            (7, Ok(2)),
            (10, Ok(3)),
            (19, Ok(4)),
            (21, Ok(5)),
            (6, Err(2)),
            (9, Err(3)),
            (18, Err(4)),
            (20, Err(5)),
            (22, Err(6)),
        ];

        cases
            .into_iter()
            .for_each(|(target, expected)| assert_eq!(bitree.binary_search(target), expected))
    }

    #[test]
    #[ntest::timeout(1000)]
    fn test_zero_array() {
        // regression: a tree containing only 0 used to loop endlessly
        let f0: BITree<usize> = BITree::from_iter([0]);
        assert_eq!(f0.prefix_sum(0), 0);
        // prefix sums: [0, 0]; searching for 1 falls past the end.
        assert_eq!(f0.binary_search(1), Err(2));
        let mut remaining = 1usize;
        assert_eq!(f0.sub_binary_search(&mut remaining), 1);
        assert_eq!(remaining, 1);
    }

    #[test]
    fn test_sub_binary_search_empty() {
        let bitree: BITree<usize> = BITree::new();
        let mut remaining = 5;
        assert_eq!(bitree.sub_binary_search(&mut remaining), 0);
        assert_eq!(remaining, 5);
        assert_eq!(bitree.binary_search(0usize), Ok(0));
        assert_eq!(bitree.binary_search(5usize), Err(1));
    }

    #[test]
    fn test_sub_binary_search_single() {
        let bitree = BITree::from_iter([7usize]);
        // prefix sums: 0, 7
        let cases: Vec<(usize, (usize, usize))> =
            vec![(0, (0, 0)), (1, (0, 1)), (7, (1, 0)), (8, (1, 1))];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_binary_search(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_binary_search_power_of_two_len() {
        // length 4 exercises a tree whose root mask equals n
        let bitree = BITree::from_iter([2usize, 3, 5, 7]);
        // prefix sums: 0, 2, 5, 10, 17
        let cases: Vec<(usize, (usize, usize))> = vec![
            (0, (0, 0)),
            (2, (1, 0)), // boundary
            (3, (1, 1)),
            (5, (2, 0)), // boundary
            (6, (2, 1)),
            (10, (3, 0)), // boundary
            (11, (3, 1)),
            (17, (4, 0)), // boundary: total sum
            (18, (4, 1)), // exceeds total
        ];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_binary_search(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_binary_search_uniform_seven() {
        // length 7 (odd, non-power-of-two) with uniform values makes expected
        // results easy to reason about: prefix_sum(k) = k.
        let bitree = BITree::from_iter([1usize; 7]);
        let cases: Vec<(usize, (usize, usize))> = vec![
            (0, (0, 0)),
            (1, (1, 0)),
            (2, (2, 0)),
            (3, (3, 0)),
            (4, (4, 0)),
            (5, (5, 0)),
            (6, (6, 0)),
            (7, (7, 0)),
            (8, (7, 1)), // exceeds total
        ];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_binary_search(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_binary_search_exceeds_total() {
        let bitree = BITree::from_iter([1usize, 6, 3, 9, 2]);
        // total sum = 21, n = 5
        let mut remaining = 100;
        assert_eq!(bitree.sub_binary_search(&mut remaining), 5);
        assert_eq!(remaining, 100 - 21);
    }

    #[test]
    fn test_push_empty() {
        let mut bitree = BITree::new();
        bitree.push(5);
        assert_eq!(bitree.inner, vec![5]);
        assert_eq!(bitree.prefix_sum(1), 5);
    }

    #[test]
    fn test_push_sequence() {
        let mut bitree = BITree::new();
        let values = [1, 6, 3, 9, 2];
        let expected_sums = vec![(1, 1), (2, 7), (3, 10), (4, 19), (5, 21)];

        for &v in values.iter() {
            bitree.push(v);
        }

        expected_sums
            .into_iter()
            .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum));
    }

    #[test]
    fn test_push_after_initialization() {
        let mut bitree = BITree::from_iter([1, 6, 3].into_iter());
        bitree.push(9);
        bitree.push(2);

        let expected_sums = vec![(1, 1), (2, 7), (3, 10), (4, 19), (5, 21)];
        expected_sums
            .into_iter()
            .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum));
    }

    #[test]
    fn test_pop_empty() {
        let mut bitree: BITree<usize> = BITree::new();
        assert_eq!(bitree.pop(), false);
    }

    #[test]
    fn test_pop_single() {
        let mut bitree = BITree::from_iter([5].into_iter());
        assert_eq!(bitree.pop(), true);
        assert!(bitree.is_empty());
    }

    #[test]
    fn test_pop_sequence() {
        let mut bitree = BITree::from_iter([1, 6, 3, 9, 2].into_iter());
        assert_eq!(bitree.pop(), true);
        assert_eq!(bitree.pop(), true);
        assert_eq!(bitree.pop(), true);

        assert_eq!(bitree.prefix_sum(1), 1);
        assert_eq!(bitree.prefix_sum(2), 7);
    }

    #[test]
    fn test_push_pop_alternating() {
        let mut bitree = BITree::new();

        bitree.push(1);
        bitree.push(6);
        assert_eq!(bitree.pop(), true);
        bitree.push(3);
        assert_eq!(bitree.pop(), true);
        bitree.push(9);
        bitree.push(2);
        assert_eq!(bitree.pop(), true);

        assert_eq!(bitree.prefix_sum(1), 1);
        assert_eq!(bitree.prefix_sum(2), 10);
    }

    #[test]
    fn test_zero_handling() {
        let mut bitree = BITree::new();
        bitree.push(0);
        bitree.push(0);
        assert_eq!(bitree.pop(), true);
        assert_eq!(bitree.prefix_sum(1), 0);
    }

    #[test]
    fn test_negative_values() {
        let mut bitree: BITree<i32> = BITree::new();
        bitree.push(-1);
        bitree.push(2);
        bitree.push(-3);

        assert_eq!(bitree.pop(), true);
        assert_eq!(bitree.prefix_sum(2), 1);
    }
}

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::ops::{AddAssign, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct BITree<T> {
    inner: Vec<T>,
}

impl<T: for<'a> AddAssign<&'a T>> FromIterator<T> for BITree<T> {
    /// Creates a new binary indexed tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths: [usize; 5] = [1, 6, 3, 9, 2];
    /// // This is how lengths binary indexed tree will look like internally
    /// let _internal: Vec<usize> = vec![1, 7, 3, 19, 2];
    /// // And this is how it can be constructed
    /// let bitree = BITree::from_iter(lengths);
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut inner: Vec<T> = iter.into_iter().collect();
        let n = inner.len();
        rebuild(&mut inner, 0..n, |p, c| *p += c);
        BITree { inner }
    }
}

impl<T: for<'a> SubAssign<&'a T>> IntoIterator for BITree<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    /// Consumes the tree and yields the original values in order.
    ///
    /// The returned iterator is `DoubleEndedIterator + ExactSizeIterator`, so it
    /// supports both forward and reverse traversal in O(1) per element after an
    /// O(n) setup that undoes the Fenwick build.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths = [1, 6, 3, 9, 2];
    /// let bitree = BITree::from_iter(lengths);
    ///
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
    fn into_iter(mut self) -> Self::IntoIter {
        let n = self.inner.len();
        rebuild(&mut self.inner, (0..n).rev(), |p, c| *p -= c);
        self.inner.into_iter()
    }
}

impl<T> BITree<T> {
    /// Creates an empty binary indexed tree.
    ///
    pub const fn new() -> Self {
        let inner = Vec::new();

        Self { inner }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline(always)]
    fn walk_prefix<F>(&self, index: usize, sum: &mut T, mut op: F)
    where
        F: FnMut(&mut T, &T),
    {
        assert!(index < self.inner.len() + 1);

        let mut current_idx = index;
        while current_idx > 0 {
            op(sum, &self.inner[current_idx - 1]);
            current_idx &= current_idx - 1;
        }
    }

    #[inline(always)]
    fn walk_update<F>(&mut self, index: usize, diff: T, mut op: F)
    where
        F: FnMut(&mut T, &T),
    {
        assert!(index < self.len());

        let mut current_idx = index;
        while let Some(value) = self.inner.get_mut(current_idx) {
            op(value, &diff);
            current_idx |= current_idx + 1;
        }
    }
}

impl<T> BITree<T> {
    /// Adds the prefix sum up until the desired index into `sum`.
    ///
    /// The prefix sum up until the zeroth element is 0, so `sum` is left unchanged.
    ///
    /// The prefix sum up until an index larger than the length is undefined, since every
    /// element after the length - 1 is undefined, therefore it will panic.
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
    pub fn add_prefix_sum(&self, index: usize, sum: &mut T)
    where
        T: for<'a> AddAssign<&'a T>,
    {
        self.walk_prefix(index, sum, |s, v| *s += v);
    }
    /// Computes the prefix sum up until the desired index, starting from `T::default()`.
    ///
    /// See [`Self::add_prefix_sum`] for the variant that accumulates into an existing value.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths = [1, 6, 3, 9, 2];
    /// let bitree = BITree::from_iter(lengths);
    ///
    /// let cases: Vec<(usize, usize)> =
    ///  vec![(0, 0), (1, 1), (2, 7), (3, 10), (4, 19), (5, 21)];
    ///
    /// cases
    ///   .into_iter()
    ///   .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum))
    /// ```
    #[inline]
    pub fn prefix_sum(&self, index: usize) -> T
    where
        T: for<'a> AddAssign<&'a T> + Default,
    {
        let mut sum = T::default();
        self.add_prefix_sum(index, &mut sum);
        sum
    }
    /// Subtracts the prefix sum up until the desired index from `sum`.
    ///
    /// The prefix sum up until the zeroth element is 0, so `sum` is left unchanged.
    ///
    /// The prefix sum up until an index larger than the length is undefined, since every
    /// element after the length - 1 is undefined, therefore it will panic.
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
    pub fn sub_prefix_sum(&self, index: usize, sum: &mut T)
    where
        T: for<'a> SubAssign<&'a T>,
    {
        self.walk_prefix(index, sum, |s, v| *s -= v);
    }
    /// Increments a given index with a difference.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths = [1, 6, 3, 9, 2];
    /// let mut bitree = BITree::from_iter(lengths);
    ///
    /// let cases: Vec<(usize, usize)> = vec![(0, 0), (1, 2), (2, 8), (3, 11), (4, 20), (5, 22)];
    ///
    /// bitree.add_at(0, 1);
    ///
    /// cases
    ///   .into_iter()
    ///   .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum))
    /// ```
    #[inline]
    pub fn add_at(&mut self, index: usize, diff: T)
    where
        T: for<'a> AddAssign<&'a T>,
    {
        self.walk_update(index, diff, |v, d| *v += d);
    }
    /// Appends a new value to the end of the Fenwick tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3].into_iter());
    /// bitree.push(9);
    ///
    /// // Check prefix sums after pushing
    /// assert_eq!(bitree.prefix_sum(1), 1);  // sum of [1]
    /// assert_eq!(bitree.prefix_sum(2), 7);  // sum of [1, 6]
    /// assert_eq!(bitree.prefix_sum(3), 10); // sum of [1, 6, 3]
    /// assert_eq!(bitree.prefix_sum(4), 19); // sum of [1, 6, 3, 9]
    /// ```
    pub fn push(&mut self, value: T)
    where
        T: for<'a> AddAssign<&'a T>,
    {
        let index = self.inner.len();
        self.inner.push(value);

        let lower_one_bits = (!index).trailing_zeros();
        let (left, right) = self.inner.split_at_mut(index);
        for i in 0..lower_one_bits {
            let child = index & !(1 << i);
            right[0] += &left[child];
        }
    }
    /// Subtracts a difference from a given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths = [1, 6, 3, 9, 2];
    /// let mut bitree = BITree::from_iter(lengths);
    ///
    /// let cases: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (2, 6), (3, 9), (4, 18), (5, 20)];
    ///
    /// bitree.sub_at(0, 1);
    ///
    /// cases
    ///   .into_iter()
    ///   .for_each(|(idx, expected_sum)| assert_eq!(bitree.prefix_sum(idx), expected_sum))
    /// ```
    #[inline]
    pub fn sub_at(&mut self, index: usize, diff: T)
    where
        T: for<'a> SubAssign<&'a T>,
    {
        self.walk_update(index, diff, |v, d| *v -= d);
    }
    /// Removes the last element from the Fenwick tree.
    ///
    /// Returns `false` if the tree is empty, and true otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let mut bitree = BITree::from_iter([1, 6, 3, 9].into_iter());
    ///
    /// assert_eq!(bitree.pop(), true);  
    /// assert_eq!(bitree.prefix_sum(3), 10);  // sum of remaining [1, 6, 3]
    ///
    /// // Can continue popping
    /// assert_eq!(bitree.pop(), true);
    /// assert_eq!(bitree.prefix_sum(2), 7);   // sum of remaining [1, 6]
    ///
    /// // Returns false when empty
    /// bitree.pop();  // removes 6
    /// bitree.pop();  // removes 1
    /// assert_eq!(bitree.pop(), false);
    /// ```
    pub fn pop(&mut self) -> bool {
        self.inner.pop().is_some()
    }
    /// Given a sum, walks the tree to find the slot containing it, subtracting the
    /// consumed segment sums from `prefix_sum` along the way.
    ///
    /// After the call, `*prefix_sum` holds the remainder — the portion of the original
    /// sum that falls strictly past the start of the returned slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let bitree = BITree::from_iter([1, 6, 3, 9, 2]);
    ///
    /// let mut remaining = 9;
    /// let idx = bitree.sub_index_of(&mut remaining);
    /// assert_eq!((idx, remaining), (2, 2));
    /// ```
    pub fn sub_index_of(&self, prefix_sum: &mut T) -> usize
    where
        T: PartialOrd + for<'a> SubAssign<&'a T>,
    {
        let n = self.inner.len();
        let mut pos = 0;
        let mut mask = most_significant_bit(n);

        while mask > 0 {
            let next = pos + mask;
            if next <= n {
                let value = &self.inner[next - 1];
                if *value < *prefix_sum {
                    pos = next;
                    *prefix_sum -= value;
                }
            }
            mask >>= 1;
        }

        pos
    }
    /// Given a sum, finds the slot in which it would be "contained" within the original
    /// array, along with the remainder — the portion of the sum that falls strictly past
    /// the start of the returned slot.
    ///
    /// If the remainder is not needed, destructure with `let (idx, _) = ...`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitree::BITree;
    ///
    /// let lengths = [1, 6, 3, 9, 2];
    /// let bitree = BITree::from_iter(lengths);
    ///
    /// let cases: Vec<(usize, (usize, usize))> = vec![(0, (0, 0)), (6, (1, 5)), (9, (2, 2)), (18, (3, 8)), (20, (4, 1))];
    ///
    /// cases
    ///   .into_iter()
    ///   .for_each(|(prefix_sum, idx)| assert_eq!(bitree.index_of(prefix_sum), idx))
    /// ```
    #[inline(always)]
    pub fn index_of(&self, mut prefix_sum: T) -> (usize, T)
    where
        T: PartialOrd + for<'a> SubAssign<&'a T>,
    {
        let index = self.sub_index_of(&mut prefix_sum);
        (index, prefix_sum)
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

#[inline(always)]
const fn most_significant_bit(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        1 << (usize::BITS - 1 - n.leading_zeros())
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
        let expected_index: Vec<usize> = vec![1, 7, 3, 19, 2];
        let actual_index = BITree::from_iter(lengths);
        assert_eq!(expected_index, actual_index.inner)
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
    fn test_index_of() {
        let lengths = [1, 6, 3, 9, 2];
        let bitree = BITree::from_iter(lengths);

        let cases: Vec<(usize, (usize, usize))> = vec![
            (0, (0, 0)),
            (6, (1, 5)),
            (9, (2, 2)),
            (18, (3, 8)),
            (20, (4, 1)),
        ];

        cases
            .into_iter()
            .for_each(|(prefix_sum, expected)| assert_eq!(bitree.index_of(prefix_sum), expected))
    }

    #[test]
    #[ntest::timeout(1000)]
    fn test_zero_array() {
        // test for a regression where index_of in an array containing only 0 would loop endlessly
        let f0: BITree<usize> = BITree::from_iter([0]);
        assert_eq!(f0.prefix_sum(0), 0);
        assert_eq!(f0.index_of(1), (1, 1));
    }

    #[test]
    fn test_sub_index_of_empty() {
        let bitree: BITree<usize> = BITree::new();
        let mut remaining = 5;
        assert_eq!(bitree.sub_index_of(&mut remaining), 0);
        assert_eq!(remaining, 5);
    }

    #[test]
    fn test_sub_index_of_single() {
        let bitree = BITree::from_iter([7usize]);
        // prefix sums: 0, 7
        let cases: Vec<(usize, (usize, usize))> =
            vec![(0, (0, 0)), (1, (0, 1)), (7, (0, 7)), (8, (1, 1))];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_index_of(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_index_of_power_of_two_len() {
        // length 4 exercises a tree whose root mask equals n
        let bitree = BITree::from_iter([2usize, 3, 5, 7]);
        // prefix sums: 0, 2, 5, 10, 17
        let cases: Vec<(usize, (usize, usize))> = vec![
            (0, (0, 0)),
            (2, (0, 2)),   // boundary: prefix_sum(1)=2 not strictly < 2
            (3, (1, 1)),
            (5, (1, 3)),   // boundary
            (6, (2, 1)),
            (10, (2, 5)),  // boundary
            (11, (3, 1)),
            (17, (3, 7)),  // boundary: total sum
            (18, (4, 1)),  // exceeds total
        ];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_index_of(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_index_of_uniform_seven() {
        // length 7 (odd, non-power-of-two) with uniform values makes expected
        // results easy to reason about: prefix_sum(k) = k.
        let bitree = BITree::from_iter([1usize; 7]);
        let cases: Vec<(usize, (usize, usize))> = vec![
            (0, (0, 0)),
            (1, (0, 1)),
            (2, (1, 1)),
            (3, (2, 1)),
            (4, (3, 1)),
            (5, (4, 1)),
            (6, (5, 1)),
            (7, (6, 1)),
            (8, (7, 1)), // exceeds total
        ];
        cases.into_iter().for_each(|(target, expected)| {
            let mut remaining = target;
            let idx = bitree.sub_index_of(&mut remaining);
            assert_eq!((idx, remaining), expected, "target={}", target);
        });
    }

    #[test]
    fn test_sub_index_of_exceeds_total() {
        let bitree = BITree::from_iter([1usize, 6, 3, 9, 2]);
        // total sum = 21, n = 5
        let mut remaining = 100;
        assert_eq!(bitree.sub_index_of(&mut remaining), 5);
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

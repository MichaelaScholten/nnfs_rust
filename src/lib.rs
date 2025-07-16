pub mod dataset;
pub mod dense;

#[cfg(test)]
const fn f64_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-15
}

#[cfg(test)]
const fn sample_equal<const SIZE: usize>(left: [f64; SIZE], right: [f64; SIZE]) -> bool {
    let mut i = 0;
    while i < SIZE {
        if !f64_equal(left[i], right[i]) {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
fn batch_equal<const SIZE: usize>(
    left: impl IntoIterator<Item = [f64; SIZE]>,
    right: impl IntoIterator<Item = [f64; SIZE]>,
) -> bool {
    left.into_iter().zip(right).all(|(left, right)| {
        left.into_iter()
            .zip(right)
            .all(|(left, right)| f64_equal(left, right))
    })
}

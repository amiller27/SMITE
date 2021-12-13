pub fn make_csr(n: usize, counts: &mut Vec<usize>) {
    for i in 1..n {
        counts[i] += counts[i - 1];
    }

    for i in (1..n + 1).rev() {
        counts[i] = counts[i - 1];
    }

    counts[0] = 0;
}

pub fn shift_csr(n: usize, counts: &mut Vec<usize>) {
    for i in (1..n + 1).rev() {
        counts[i] = counts[i - 1];
    }
    counts[0] = 0;
}

pub fn bucket_sort_keys_increasing(
    max: usize,
    keys: &Vec<usize>,
    tperm: Vec<usize>,
) -> Vec<usize> {
    let mut counts = vec![0; max + 2];

    for &key in keys {
        counts[key] += 1;
    }

    make_csr(max + 1, &mut counts);

    let mut perm = vec![0; keys.len()];

    for i in 0..keys.len() {
        perm[counts[keys[tperm[i]]]] = tperm[i];
        counts[keys[tperm[i]]] += 1;
    }

    perm
}

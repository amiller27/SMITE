pub fn make_csr(max: usize, counts: &mut Vec<usize>) {
    for i in 1..max + 1 {
        counts[i] += counts[i - 1];
    }

    for i in (1..max + 2).rev() {
        counts[i] = counts[i - 1];
    }

    counts[0] = 0;
}

pub fn shift_csr(max: usize, counts: &mut Vec<usize>) {
    for i in (1..max + 1).rev() {
        counts[i] = counts[i - 1];
    }
    counts[0] = 0;
}

pub fn bucket_sort_keys_increasing(max: usize, keys: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut counts = vec![0; max + 2];

    for key in keys {
        counts[*key] += 1;
    }

    make_csr(max, &mut counts);

    let mut tperm = vec![0; keys.len()];
    let mut perm = vec![0; keys.len()];

    for i in 0..keys.len() {
        perm[counts[keys[tperm[i]]]] = tperm[i];
        counts[keys[tperm[i]]] += 1;
    }

    (tperm, perm)
}

use std::cmp::Ord;

const QSORT_MAX_THRESH: usize = 8;

const QSORT_STACK_SIZE: usize = 64;
struct FastStack {
    stack: [(usize, usize); QSORT_STACK_SIZE],
    top: usize,
}

impl FastStack {
    fn push(&mut self, lo: usize, hi: usize) {
        self.stack[self.top] = (lo, hi);
        self.top += 1;
    }

    fn pop(&mut self) -> (usize, usize) {
        self.top -= 1;
        self.stack[self.top]
    }

    fn not_empty(&self) -> bool {
        self.top != 0
    }

    fn new() -> FastStack {
        FastStack {
            stack: [(0, 0); QSORT_STACK_SIZE],
            top: 1,
        }
    }
}

pub fn quicksort<T>(arr: &mut Vec<T>)
where
    T: Ord + Clone,
{
    let base = 0;
    let elems = arr.len();

    if elems < 1 {
        return;
    }

    if elems > QSORT_MAX_THRESH {
        let mut lo = base;
        let mut hi = lo + elems - 1;

        let mut stack = FastStack::new();

        while stack.not_empty() {
            let mut mid = lo + ((hi - lo) >> 1);

            // Select median value from among LO, MID, and HI. Rearrange
            // LO and HI so the three values are sorted. This lowers the
            // probability of picking a pathological pivot value and
            // skips a comparison for both the LEFT_PTR and RIGHT_PTR in
            // the while loops.
            if arr[mid] < arr[lo] {
                arr.swap(mid, lo);
            }

            if arr[hi] < arr[mid] {
                arr.swap(mid, hi);
                if arr[mid] < arr[lo] {
                    arr.swap(mid, lo);
                }
            }

            let mut left_ptr = lo + 1;
            let mut right_ptr = hi - 1;

            loop {
                while arr[left_ptr] < arr[mid] {
                    left_ptr += 1;
                }

                while arr[mid] < arr[right_ptr] {
                    right_ptr -= 1;
                }

                if left_ptr < right_ptr {
                    arr.swap(left_ptr, right_ptr);

                    if mid == left_ptr {
                        mid = right_ptr;
                    } else if mid == right_ptr {
                        mid = left_ptr;
                    }

                    left_ptr += 1;
                    right_ptr -= 1;
                } else if left_ptr == right_ptr {
                    left_ptr += 1;
                    right_ptr -= 1;
                    break;
                }

                if left_ptr > right_ptr {
                    break;
                }
            }

            if right_ptr - lo <= QSORT_MAX_THRESH {
                if hi - left_ptr <= QSORT_MAX_THRESH {
                    // ignore both small partitions
                    (lo, hi) = stack.pop();
                } else {
                    // ignore small left partition
                    lo = left_ptr;
                }
            } else if hi - left_ptr <= QSORT_MAX_THRESH {
                // ignore small right partition
                hi = right_ptr;
            } else if right_ptr - lo > hi - left_ptr {
                // push larger left partition indices
                stack.push(lo, right_ptr);
                lo = left_ptr;
            } else {
                // push larger right partition indices
                stack.push(left_ptr, hi);
                hi = right_ptr;
            }
        }
    }

    // Once the BASE array is partially sorted by quicksort the rest
    // is completely sorted using insertion sort, since this is efficient
    // for partitions below MAX_THRESH size. BASE points to the
    // beginning of the array to sort, and END_PTR points at the very
    // last element in the array (*not* one beyond it!).
    let end_ptr = base + elems - 1;
    let mut tmp_ptr = base;

    let mut thresh = base + QSORT_MAX_THRESH;
    if thresh > end_ptr {
        thresh = end_ptr;
    }

    // Find smallest element in first threshold and place it at the
    // array's beginning.  This is the smallest array element,
    // and the operation speeds up insertion sort's inner loop.

    for run_ptr in (tmp_ptr + 1)..(thresh + 1) {
        if arr[run_ptr] < arr[tmp_ptr] {
            tmp_ptr = run_ptr;
        }
    }

    if tmp_ptr != base {
        arr.swap(tmp_ptr, base);
    }

    // Insertion sort, running from left-hand-side
    // up to right-hand-side.

    let mut run_ptr = base + 1;
    loop {
        run_ptr += 1;
        if run_ptr > end_ptr {
            break;
        }

        tmp_ptr = run_ptr - 1;
        while arr[run_ptr] < arr[tmp_ptr] {
            tmp_ptr -= 1;
        }

        tmp_ptr += 1;
        if tmp_ptr != run_ptr {
            let mut trav = run_ptr + 1;
            loop {
                trav -= 1;
                if trav < run_ptr {
                    break;
                }
                let hold = arr[trav].clone();

                let mut hi = trav;
                let mut lo = trav;
                loop {
                    lo -= 1;
                    if lo < tmp_ptr {
                        break;
                    }
                    arr[hi] = arr[lo].clone();
                    hi = lo;
                }
                arr[hi] = hold.clone();
            }
        }
    }
}

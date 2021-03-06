use std::os::raw::c_char;

#[link(name = "metis_test")]
extern "C" {
    #[allow(dead_code)]
    fn run_metis_ffi(mat_name: *const c_char, ordering: *mut i64, n_ordering: i64) -> bool;
}

#[allow(dead_code)]
pub fn run_metis(mat_name: &str, size: usize) -> Vec<usize> {
    let mut metis_ordering = vec![0; size];

    let c_mat_name = std::ffi::CString::new(mat_name).unwrap();

    unsafe {
        let success = run_metis_ffi(
            c_mat_name.as_ptr(),
            metis_ordering.as_mut_ptr(),
            size as i64,
        );

        if !success {
            panic!("METIS failed");
        }
    }

    metis_ordering.iter().map(|&x| x as usize).collect()
}

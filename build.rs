fn main() {
    // Directory containing metis_test.so
    println!("cargo:rustc-link-search=native=/home/aaron/metis-cpp/test");
    println!("cargo:rustc-link-search=native=/home/aaron/metis-cpp/METIS/build/libmetis");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=metis");
}
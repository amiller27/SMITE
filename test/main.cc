#include <fmt/format.h>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

void run_metis(const char *const mat_name, int64_t *const ordering,
               const int64_t n_ordering);

// This is kinda dumb, it's all to avoid passing owned pointers out through the
// C FFI
size_t get_size(const std::string &mat_name) {
  Eigen::SparseMatrix<double> A;
  Eigen::loadMarket(
      A, fmt::format("/home/aaron/matrices/{}/{}.mtx", mat_name, mat_name));
  return A.rows();
}

int main() {
  const std::string mat_name = "494_bus";

  std::vector<int64_t> ordering(get_size(mat_name));
  run_metis(mat_name.c_str(), ordering.data(), ordering.size());
}
// clang-format off
#include <iostream>
// clang-format on

#include <Eigen/MetisSupport>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <GKlib.h>

void run_metis(const char *const mat_name, int64_t *const ordering,
               const int64_t n_ordering) {
  const auto rng_trace_filename =
      fmt::format("/home/aaron/rng_traces/{}.txt", mat_name);
  gk_random_trace_filename = rng_trace_filename.c_str();

  Eigen::SparseMatrix<double> A;
  Eigen::loadMarket(
      A, fmt::format("/home/aaron/matrices/{}/{}.mtx", mat_name, mat_name));
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int64_t> perm{};
  Eigen::MetisOrdering<Eigen::SparseMatrix<double>::StorageIndex>()(A, perm);

  gk_random_flush_trace();

  if (n_ordering != perm.rows()) {
    throw std::runtime_error(fmt::format("Ordering size should be {}, got {}",
                                         perm.rows(), n_ordering));
  }

  for (int i = 0; i < n_ordering; i++) {
    ordering[i] = perm.indices()[i];
  }
}

extern "C" bool test(const char *const mat_name, int64_t *const ordering,
                     const int64_t n_ordering) {
  try {
    run_metis(mat_name, ordering, n_ordering);
    return true;
  } catch (const std::exception &ex) {
    fmt::print("Caught exception:\n{}\n", ex.what());
    return false;
  } catch (...) {
    fmt::print("Caught unknown exception\n");
    return false;
  }
}
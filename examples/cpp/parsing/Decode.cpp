#include <Eigen/Dense>
#include "FactorTree.h"
#include "Utils.h"
#include "logval.h"

// Define a matrix of doubles using Eigen.
typedef LogVal<double> LogValD;
namespace Eigen {
  typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
}

namespace AD3
{
  // decoder for an arc factored parser; it invokes the matrix-tree theorem.
  // index is a n x (n - 1) matrix mapping (head, modifier) to either -1 (if
  // it doesn't exist) or its index in the scores vector.
  void DecodeMatrixTree(const vector<vector<int> > &index, const vector<Arc*> &arcs,
                        const vector<double> &scores,
                        vector<double> *predicted_output,
                        double *log_partition_function, double *entropy) {
    int length = index.size();

    // Matrix for storing the potentials.
    Eigen::MatrixXlogd potentials(length, length);
    // Kirchhoff matrix.
    Eigen::MatrixXlogd kirchhoff(length - 1, length - 1);

    // Compute an offset to improve numerical stability. This is a constant that
    // is subtracted from all scores.
    int num_arcs = arcs.size();
    double constant = 0.0;
    for (int r = 0; r < num_arcs; ++r) {
      constant += scores[r];
    }
    constant /= static_cast<double>(num_arcs);

    // Set the potentials.
    for (int h = 0; h < length; ++h) {
      for (int m = 0; m < length; ++m) {
      potentials(m, h) = LogValD::Zero();
      int r = index[h][m];
      if (r >= 0) {
        potentials(m, h) = LogValD(scores[r] - constant, false);
      }
      }
    }

    // Set the Kirchhoff matrix.
    for (int h = 0; h < length - 1; ++h) {
      for (int m = 0; m < length - 1; ++m) {
      kirchhoff(h, m) = -potentials(m + 1, h + 1);
      }
    }
    for (int m = 1; m < length; ++m) {
      LogValD sum = LogValD::Zero();
      for (int h = 0; h < length; ++h) {
      sum += potentials(m, h);
      }
      kirchhoff(m - 1, m - 1) = sum;
    }

    // Inverse of the Kirchoff matrix.
    Eigen::FullPivLU<Eigen::MatrixXlogd> lu(kirchhoff);
    Eigen::MatrixXlogd inverted_kirchhoff = lu.inverse();
    *log_partition_function = lu.determinant().logabs() +
        constant * (length - 1);

    Eigen::MatrixXlogd marginals(length, length);
    for (int h = 0; h < length; ++h) {
      marginals(0, h) = LogValD::Zero();
    }
    for (int m = 1; m < length; ++m) {
      marginals(m, 0) = potentials(m, 0) * inverted_kirchhoff(m - 1, m - 1);
      for (int h = 1; h < length; ++h) {
      marginals(m, h) = potentials(m, h) *
          (inverted_kirchhoff(m - 1, m - 1) - inverted_kirchhoff(m - 1, h - 1));
      }
    }

    // Compute the entropy.
    predicted_output->resize(num_arcs);
    *entropy = *log_partition_function;
    for (int r = 0; r < num_arcs; ++r) {
      int h = arcs[r]->head();
      int m = arcs[r]->modifier();
      if (marginals(m, h).signbit()) {
      if (!NEARLY_ZERO_TOL(marginals(m, h).as_float(), 1e-6)) {
        // LOG(INFO) << "Marginals truncated to zero (" << marginals(m, h).as_float() << ")";
        cerr << "Marginals truncated to zero (" << marginals(m, h).as_float() << ")";
      }
      // CHECK(!std::isnan(marginals(m, h).as_float()));
      } else if (marginals(m, h).logabs() > 0) {
      if (!NEARLY_ZERO_TOL(marginals(m, h).as_float() - 1.0, 1e-6)) {
        // LOG(INFO) << "Marginals truncated to one (" << marginals(m, h).as_float() << ")";
        cerr << "Marginals truncated to one (" << marginals(m, h).as_float() << ")";
      }
      }
      (*predicted_output)[r] = marginals(m, h).as_float();
      *entropy -= (*predicted_output)[r] * scores[r];
    }
    if (*entropy < 0.0) {
      if (!NEARLY_ZERO_TOL(*entropy, 1e-6)) {
        // LOG(INFO) << "Entropy truncated to zero (" << *entropy << ")";
        cerr << "Entropy truncated to zero (" << *entropy << ")";
      }
      *entropy = 0.0;
    }
  }

} // AD3

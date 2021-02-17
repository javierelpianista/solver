#include <vector>
#include <iostream>
#include <functional>
#include <solver.hpp>
#include <math.h>
#include <complex>

#include <Eigen/Dense>

#include <boost/multiprecision/eigen.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

typedef boost::multiprecision::mpc_complex_500 C;
typedef boost::multiprecision::mpfr_float_500 R;

int main() {
    std::function<C(Eigen::Matrix<C,2,1>&)> f, g;

    f = [] (Eigen::Matrix<C,2,1> &x) -> C {
        return x(0)*x(1) - C(6)/10;
    };

    g = [] (Eigen::Matrix<C,2,1> &x) -> C {
        return x(0)*x(0) + x(1)*x(1) - C(2);
    };

    std::vector<decltype(f)> vf;

    vf.push_back(f);
    vf.push_back(g);

    Eigen::Matrix<C,2,1> x0, x;

    x0(0) = 1;
    x0(1) = 1;

    auto s = solver::Solver2<C,R,2>(vf);
    s.set_h(R("1e-200"));
    s.set_tol(R("1e-100"));
    x = s.solve(x0);

    return 0;
}

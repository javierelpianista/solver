#ifndef RICPAD_SOLVER
#define RICPAD_SOLVER

#include <iostream>
#include <vector>
#include <math.h>

#include <Eigen/Dense>

#include <differentiate.hpp>

using std::cout;
using std::endl;

namespace solver { 
template<typename num_t>
class Solver {
    private:
        // Function
        const std::function<num_t(num_t)> f_;
        // Derivative of the function
        const std::function<num_t(num_t)> df_;
        // Step size
        num_t h_;
        // Tolerance of the result
        num_t tol_;

    public: 
        // Construct a Solver object from the function and its derivative
        Solver( 
            const std::function<num_t(num_t)> f,
            const std::function<num_t(num_t)> df
            ) :
                
            f_(f), df_(df)
            {
                h_   = std::sqrt(std::numeric_limits<num_t>::epsilon());
                tol_ = std::sqrt(h_);
            }

        // Solve f = 0 for the initial value of x0
        num_t solve(num_t x0) {
            num_t desv, fx, dfx, x = x0, xold;

            desv = 2*tol_;

            while ( desv > tol_ ) {
                fx  = f_(x);
                dfx = df_(x);

                xold = x;
                x = x - fx/dfx;

                std::cout << x << "\n";

                desv = std::abs(x - xold);
            }

            return x;
        }
};

// Solver for N equations using a real type R
template <
    typename C, // complex number type
    typename R, // real number type (for tolerance parameters, etc.)
    int N       // number of equations and unknowns
>
class Solver2 {
    private:
        // Vector with the functions
        const std::vector<std::function<C(Eigen::Matrix<C,N,1>&)>> f_;
        //Eigen::Matrix<C,N,1> x0;
        // Accepted difference between two iterations
        R tol_; 
        // Numerical value for the differentiation step (only relevant when
        // using numerical differentiation)
        R h_;
        // Maximum number of iterations
        int maxiter_ = 100;

    public:
        //----------------------------------------------------------------------
        // Construct a Solver2 object from a vector of functions. 
        // Differentiation is performed numerically.
        Solver2(
            const std::vector<std::function<C(Eigen::Matrix<C,N,1>&)>> &f
            ) :
            
            f_(f) {
                //h_   = std::sqrt(std::numeric_limits<R>::epsilon());
                h_   = 1e-12;
                tol_ = 1e-8;
            };

        //----------------------------------------------------------------------
        // Setters
        void set_h(R h) {h_ = h;};
        void set_tol(R tol) {tol_ = tol;};
        void set_maxiter(int maxiter) {maxiter_ = maxiter;};

        //----------------------------------------------------------------------
        // Solve for f using x0 as initial value
        Eigen::Matrix<C,N,1> solve(
            Eigen::Matrix<C,N,1> x0
            ) 
        {

            std::cout << "h = " << h_ << "\n"
                << "tol = " << tol_ << "\n\n";

            using solver::differentiate;
            Eigen::Matrix<C, N, N> jacobian, inv_jacobian;
            Eigen::Matrix<C, N, 1> x(x0), xold;
            R desv = tol_ + 1;

            int niter = 0;

            C val;

            while ( desv > tol_ ) {
                for ( int i = 0; i < x.size(); i++ ) {
                    for ( int j = 0; j < x.size(); j++ ) {
                        val = differentiate<C, N>(f_[i], x, j, h_);
                        jacobian(i, j) = std::move(val);
                    }
                }

                inv_jacobian = jacobian.inverse();
                
                Eigen::Matrix<C, N, 1> F;

                for ( int i = 0; i < N; i++ ) F(i) = f_[i](x); 
                /*
                for ( int i = 0; i < N; i++ ) {
                    std::cout << std::setprecision(15) << x(i) << " ";
                }
                std::cout << "\n";
                */
                xold = x;
                x = x - inv_jacobian * F;

                desv = (x - xold).norm();
                std::cout << "desv = " << desv << "\n";

                if ( niter++ > maxiter_ ) {
                    std::cout << "Maximum number of iterations reached.\n";
                    return x;
                }
            }

            return x;
        };

        /* Construct a Solver2 object from a function and its derivative
        Solver2(
            const std::function<R(R)> f,
            const std::function<R(R)> df
            )
        */
};

}; // namespace solver

#endif

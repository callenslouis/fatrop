#include <vector>
#include <iostream>
#include <chrono>
#include <random>

#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"

using namespace fatrop;

int main(){
    // setup random dimensions
    int nb_runs = 10;
    std::vector<int> nu(nb_runs);
    std::vector<int> nx(nb_runs);
    std::vector<int> ng(nb_runs);
    int max_val = 5;
    for (int i = 0; i < nb_runs; ++i) {
        nu[i] = rand() % (max_val+1); // Random number of control inputs between 1 and 100
        nx[i] = rand() % (max_val+1); // Random number of states between 1 and 100
        ng[i] = rand() % (max_val+1); // Random number of constraints between 1 and 100
    }

    // allocate random matrices
    std::vector<MatRealAllocated> A_full;
    std::vector<PermutationMatrix> Pl_full;
    std::vector<PermutationMatrix> Pr_full;
    std::vector<MatRealAllocated> A_blocked;
    std::vector<PermutationMatrix> Pl_blocked1;
    std::vector<PermutationMatrix> Pr_blocked1;
    std::vector<PermutationMatrix> Pl_blocked2;
    std::vector<PermutationMatrix> Pr_blocked2;
    MatRealAllocated A_copy(2*max_val, 2*max_val);
    MatRealAllocated A_verification(2*max_val, 2*max_val);
    MatRealAllocated A_verification_T(2*max_val, 2*max_val);
    MatRealAllocated L(2*max_val, 2*max_val);
    MatRealAllocated U(2*max_val, 2*max_val);

    A_full.reserve(nb_runs);
    Pl_full.reserve(nb_runs);
    Pr_full.reserve(nb_runs);
    A_blocked.reserve(nb_runs);
    Pl_blocked1.reserve(nb_runs);
    Pr_blocked1.reserve(nb_runs);
    Pl_blocked2.reserve(nb_runs);
    Pr_blocked2.reserve(nb_runs);
    for (int k = 0; k < nb_runs; k++){
        double progress = (double)(k+1) / nb_runs * 100.0;
        std::cout << "allocating ... " << std::setw(4) << std::setprecision(3) << progress << "%\r" << std::flush;
        int m = ng[k] + nx[k];
        int n = nu[k] + nx[k];
        A_full.emplace_back(n, m);
        A_blocked.emplace_back(n, m);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                A_full[k](i,j) = rand() / double(RAND_MAX);
                if (j >= ng[k] || i < nu[k]){ // transposed
                    A_blocked[k](i,j) = A_full[k](i,j);
                } else {
                    A_blocked[k](i,j) = 0.0; // zero out the top-left block
                }
            }
        }
        Pl_full.emplace_back(m);
        Pr_full.emplace_back(n);
        Pl_blocked1.emplace_back(m);
        Pr_blocked1.emplace_back(n);
        Pl_blocked2.emplace_back(m);
        Pr_blocked2.emplace_back(n);
    }
    std::cout << std::flush << std::endl;
    std::cout << "allocations done" << std::endl;

    std::vector<double> time_full(nb_runs);
    std::vector<double> time_blocked(nb_runs);

    // perform full lu timings
    int r;
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        std::cout << "running full LU ... " << std::setw(4) << std::setprecision(3) << progress << "%\r" << std::flush;

        // store current matrix
        blasfeo_dgese(A_copy.m(), A_copy.n(), 0, &A_copy.mat(), 0, 0);
        blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
        blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
        blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);
        blasfeo_dgecp(nu[i]+nx[i], ng[i]+nx[i], &A_full[i].mat(), 0, 0, &A_copy.mat(), 0, 0);

        // perform lu
        auto start = std::chrono::high_resolution_clock::now();
        fatrop_lu_fact_transposed(ng[i]+nx[i], nu[i]+nx[i], nu[i]+nx[i], r, &A_full[i].mat(), Pl_full[i], Pr_full[i], 1e-5);
        auto end = std::chrono::high_resolution_clock::now();
        time_full[i] = std::chrono::duration<double>(end - start).count();

        // verify     
        // extract L
        for (int row = 0; row < ng[i]+nx[i]; row++){
            for (int col = 0; col < ng[i]+nx[i]; col++){
                if (row > col /*&& col < nu[i]+nx[i]*/){
                    L(row,col) = A_full[i](col,row);
                } else if (row == col){
                    L(row,col) = 1.0;
                } else {
                    L(row,col) = 0.0;
                }
            }
        }
        // extract U
        for (int row = 0; row < nu[i]+nx[i]; row++){
            for (int col = 0; col < nu[i]+nx[i]; col++){
                if (row <= col /*&& row < ng[i]+nx[i]*/){
                    U(row,col) = A_full[i](col,row);
                } else {
                    U(row,col) = 0.0;
                }
            }
        }
        // compute L*U
        blasfeo_dgemm_nn(ng[i]+nx[i], nu[i]+nx[i], nu[i]+nx[i], 1.0, 
                         &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                         &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

        std::cout << "A_verification before permutations:\n" << A_verification << std::endl;
        // apply permutations
        std::cout << "Pl:\n" << Pl_full[i] << std::endl;
        std::cout << "Pr:\n" << Pl_full[i] << std::endl;
        // Pl_full[i].apply_on_rows(r, &A_verification.mat());
        // Pr_full[i].apply_on_cols(r, &A_verification.mat());
        Pl_full[i].apply_on_cols(r, &A_copy.mat());
        Pr_full[i].apply_on_rows(r, &A_copy.mat());

        // transpose A_verification
        // for (int row = 0; row < ng[i]+nx[i]; row++){
        //     for (int col = row+1; col < nu[i]+nx[i]; col++){
        //         double temp = A_verification(row,col);
        //         A_verification(row,col) = A_verification(col,row);
        //         A_verification(col,row) = temp;
        //     }
        // }
        blasfeo_dgetr(ng[i]+nx[i], nu[i]+nx[i], &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);

        std::cout << "A_verification after transpose:\n" << A_verification_T << std::endl;
        

        // check that A_copy and A_verification are close
        double max_diff = 0.0;
        for (int row = 0; row < nu[i]+nx[i]; row++){
            for (int col = 0; col < ng[i]+nx[i]; col++){
                double diff = std::abs(A_copy(row,col) - A_verification_T(row,col));
                if (diff > max_diff){
                    max_diff = diff;
                }
            }
        }
        if (max_diff > 1e-4){
            std::cout << "\nLU factorization verification failed for run " << i+1 << "/" << nb_runs << std::endl;
            std::cout << "A is an " << (ng[i]+nx[i]) << "x" << (nu[i]+nx[i]) << " matrix" << std::endl;
            std::cout << "Max difference: " << max_diff << std::endl;
            std::cout << "A_copy:\n" << A_copy << std::endl;
            std::cout << "A_verification:\n" << A_verification_T << std::endl;
            std::cout << "L:\n" << L << std::endl;
            std::cout << "U:\n" << U << std::endl;
            return -1;
        }

    }
    std::cout << "\nfull LU done" << std::endl;

    // perform blocked lu timings
    int r1 = 0; 
    int r2 = 0;
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        std::cout << "running blocked LU ... " << std::setw(4) << std::setprecision(3) << progress << "%\r" << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "\n\n\n\nA before:\n" << A_blocked[i] << std::endl;
        fatrop_lu_fact_transposed(ng[i], nu[i], nu[i], r1, &A_blocked[i].block(nu[i], ng[i], 0, 0).mat(), Pl_blocked1[i], Pr_blocked1[i], 1e-5);
        std::cout << "nu: " << nu[i] << ", ng: " << ng[i] << ", nx: " << nx[i] << std::endl;
        std::cout << "A intermediate:\n" << A_blocked[i] << std::endl;
        std::cout << "block:\n" << A_blocked[i].block(nx[i], nx[i], nu[i], ng[i]) << std::endl;
        std::cout << "block.mat():" << std::endl;
        fatrop_lu_fact_transposed(nx[i], nx[i], nx[i], r2, &A_blocked[i].block(nx[i], nx[i], nu[i], ng[i]).mat(), Pl_blocked2[i], Pr_blocked2[i], 1e-5);
        std::cout << "A after:\n" << A_blocked[i] << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        time_blocked[i] = std::chrono::duration<double>(end - start).count();
    }
    std::cout << "\nblocked LU done" << std::endl;
}
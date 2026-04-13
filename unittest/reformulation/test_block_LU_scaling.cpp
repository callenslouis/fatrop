#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>

#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"

using namespace fatrop;

void extract_L(const MatRealAllocated &LU, MatRealAllocated &L, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < m; row++){
        for (int col = 0; col < m; col++){
            if (row > col /*&& col < nu[i]+nx[i]*/){
                L(bi+row,bj+col) = LU(ai+col,aj+row);
            } else if (row == col){
                L(bi+row,bj+col) = 1.0;
            } else {
                L(bi+row,bj+col) = 0.0;
            }
        }
    }
}

void extract_U(const MatRealAllocated &LU, MatRealAllocated &U, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            if (row <= col /*&& row < ng[i]+nx[i]*/){
                U(bi+row,bj+col) = LU(ai+col,aj+row);
            } else {
                U(bi+row,bj+col) = 0.0;
            }
        }
    }
}

bool verify_lu(const MatRealAllocated &LU, const MatRealAllocated &A_original, MatRealAllocated &A_verification, 
               MatRealAllocated &A_verification_T, 
               MatRealAllocated &L, MatRealAllocated &U, 
               PermutationMatrix &Pl, PermutationMatrix &Pr, int rank, 
               int m, int n, int ai=0, int aj=0){
    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
    blasfeo_dgese(A_verification_T.m(), A_verification_T.n(), 0, &A_verification_T.mat(), 0, 0);
    blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
    blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);

    if (ai != 0 || aj != 0){
        // gecp(m, n, LU, ai, aj, LU, 0, 0);
        gecp(n, m, A, aj, ai, A, 0, 0);
    }

    // extract L and U
    extract_L(LU, L, m, n, ai, aj);
    extract_U(LU, U, m, n, ai, aj);

    // compute L*U
    blasfeo_dgemm_nn(m, n, n, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // apply permutations
    Pl.apply_on_cols(rank, &A.mat());
    Pr.apply_on_rows(rank, &A.mat());

    // transpose A_verification
    blasfeo_dgetr(m, n, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);  

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < n; row++){
        for (int col = 0; col < m; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }
    if (max_diff > 1e-4){
        std::cout << "\nLU factorization verification failed" << std::endl;
        std::cout << "A is an " << (m) << "x" << (n) << " matrix" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "A_copy:\n" << LU << std::endl;
        std::cout << "A_verification:\n" << A_verification_T << std::endl;
        std::cout << "L:\n" << L << std::endl;
        std::cout << "U:\n" << U << std::endl;
        return false;
    }
    return true;
}

bool verify_blocked_lu(const MatRealAllocated& LU, const MatRealAllocated& B_tilde, 
                       const MatRealAllocated& A_original, MatRealAllocated& A_verification, 
                       MatRealAllocated& A_verification_T, MatRealAllocated& L, MatRealAllocated& U, 
                       PermutationMatrix& Pl1, PermutationMatrix& Pr1, int rank1,
                       PermutationMatrix& Pl2, PermutationMatrix& Pr2, int rank2,
                       int ng, int nu, int nx){
    PermutationMatrix Pl_total(ng+nx);
    PermutationMatrix Pr_total(nu+nx);
    for (int i = 0; i < nx; i++){
        Pl_total[i] = Pl1[i];
        Pr_total[i] = Pr1[i];
    }
    for (int i = 0; i < rank2; i++){
        Pl_total[nx + i] = Pl2[i] + nx;
        Pr_total[nx + i] = Pr2[i] + nx;
    }

    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
    blasfeo_dgese(A_verification_T.m(), A_verification_T.n(), 0, &A_verification_T.mat(), 0, 0);
    blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
    blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);


    // extract L2 and U2
    extract_L(LU, L, ng, ng, nx, nx, nx, nx);
    extract_U(LU, U, ng, nu, nx, nx, nx, nx);

    // extract L1 and U1
    extract_L(LU, L, nx, nx, 0, 0);
    extract_U(LU, U, nx, nx, 0, 0);

    // copy B_tilde
    getr(nx, ng, B_tilde, 0, 0, L, nx, 0);

    // compute L*U
    blasfeo_dgemm_nn(ng+nx, nu+nx, ng+nx, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // apply permutations
    int r = nx + rank2;
    Pl_total.apply_on_cols(r, &A.mat());
    Pr_total.apply_on_rows(r, &A.mat());

    // transpose A_verification
    blasfeo_dgetr(ng+nx, nu+nx, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < nu+nx; row++){
        for (int col = 0; col < ng+nx; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }

    if (max_diff > 1e-4){
        std::cout << "\nBlocked LU factorization verification failed" << std::endl;
        std::cout << "A is an " << (ng+nx) << "x" << (nu+nx) << " matrix" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "A_original:\n" << A_original << std::endl;
        std::cout << "A_copy:\n" << A << std::endl;
        std::cout << "A_verification:\n" << A_verification_T << std::endl;
        std::cout << "L:\n" << L << std::endl;
        std::cout << "U:\n" << U << std::endl;
        return false;
    }

    return true;
}


bool verify_blocked_lu_new(const MatRealAllocated& LU, 
        const MatRealAllocated& A_original, MatRealAllocated& A_verification, 
        MatRealAllocated& A_verification_T, MatRealAllocated& L, MatRealAllocated& U, 
        PermutationMatrix& Pl1, PermutationMatrix& Pr1, int rank1,
        PermutationMatrix& Pl_rank,
        PermutationMatrix& Pl2, PermutationMatrix& Pr2, int rank2,
        int ng, int nu, int nx){
    // PermutationMatrix Pl_total(ng+nx);
    // PermutationMatrix Pr_total(nu+nx);
    // for (int i = 0; i < nx; i++){
    //     Pl_total[i] = Pl1[i];
    //     Pr_total[i] = Pr1[i];
    // }
    // for (int i = 0; i < rank2; i++){
    //     Pl_total[nx + i] = Pl2[i] + nx;
    //     Pr_total[nx + i] = Pr2[i] + nx;
    // }

    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
    blasfeo_dgese(A_verification_T.m(), A_verification_T.n(), 0, &A_verification_T.mat(), 0, 0);
    blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
    blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);


    // // extract L2 and U2
    // extract_L(LU, L, ng, ng, nx, nx, nx, nx);
    // extract_U(LU, U, ng, nu, nx, nx, nx, nx);

    // // extract L1 and U1
    // extract_L(LU, L, nx, nx, 0, 0);
    // extract_U(LU, U, nx, nx, 0, 0);

    extract_L(LU, L, ng+nx, ng+nx, 0, 0);
    extract_U(LU, U, ng+nx, nu+nx, 0, 0);

    // // copy B_tilde
    // getr(nx, ng, B_tilde, 0, 0, L, nx, 0);

    // compute L*U
    blasfeo_dgemm_nn(ng+nx, nu+nx, ng+nx, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // // apply permutations
    // int r = nx + rank2;
    // Pl_total.apply_on_cols(r, &A.mat());
    // Pr_total.apply_on_rows(r, &A.mat());
    Pl1.apply_on_cols(rank1, &A.mat());
    Pl_rank.apply_on_cols(nx+ng, &A.mat());
    Pl2.apply_on_cols(rank2, &A.mat(), rank1);

    Pr1.apply_on_rows(rank1, &A.mat());
    Pr2.apply_on_rows(rank2, &A.mat(), rank1);

    // transpose A_verification
    blasfeo_dgetr(ng+nx, nu+nx, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < nu+nx; row++){
        for (int col = 0; col < ng+nx; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }

    if (max_diff > 1e-4){
        std::cout << "\nBlocked LU factorization verification failed" << std::endl;
        std::cout << "A is an " << (ng+nx) << "x" << (nu+nx) << " matrix" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "A_original:\n" << A_original << std::endl;
        std::cout << "A_copy:\n" << A << std::endl;
        std::cout << "A_verification:\n" << A_verification_T << std::endl;
        std::cout << "L:\n" << L << std::endl;
        std::cout << "U:\n" << U << std::endl;
        return false;
    }

    return true;
}

void write_np_matrix(const MatRealAllocated& M, int m, int n, std::string name){
    std::ostream& o = std::cout;

    o << name << " = np.array([\n";
    for (int i = 0; i < m; i++){
        o << "    [";
        for (int j = 0; j < n; j++){
            o << std::setprecision(12) << M(i,j);
            if (j < n-1){
                o << ", ";
            }
        }
        o << "]";
        if (i < m-1){
            o << ",\n";
        } else {
            o << "\n";
        }
    } 
    o << "])\n";
    o << std::flush;    
}

int main(){

    // PermutationMatrix p1(3);
    // PermutationMatrix p2(3);
    // p1[0] = 1;
    // // p2[1] = 2;
    // std::cout << "before:" << std::endl;
    // std::cout << "p1: " << p1 << std::endl;
    // std::cout << "p2: " << p2 << std::endl;

    // p2.apply(3, p1, 0);
    // std::cout << "after:" << std::endl;
    // std::cout << "p1: " << p1 << std::endl;
    // std::cout << "p2: " << p2 << std::endl;
    // return 0;


    // setup random dimensions
    int nb_runs = 100;//1000000;
    bool verify = true;
    std::string file_name = "blocked_lu_timings.csv";
    bool write_csv = false;

    std::vector<int> nu(nb_runs);
    std::vector<int> nx(nb_runs);
    std::vector<int> ng(nb_runs);
    int max_val = 4;//25;
    for (int i = 0; i < nb_runs; ++i) {
        nu[i] = rand() % (max_val+1); // Random number of control inputs between 1 and 100
        nx[i] = rand() % (max_val+1); // Random number of states between 1 and 100
        ng[i] = rand() % (max_val+1); // Random number of constraints between 1 and 100

        // nu[i] = 1;
        // nx[i] = 2;
        // ng[i] = 1;
    }

    // allocate random matrices
    std::vector<MatRealAllocated> A_full;
    std::vector<PermutationMatrix> Pl_full;
    std::vector<PermutationMatrix> Pr_full;
    std::vector<MatRealAllocated> A_blocked;
    std::vector<PermutationMatrix> Pl_blocked1;
    std::vector<PermutationMatrix> Pr_blocked1;
    std::vector<PermutationMatrix> Pl_rank;
    std::vector<PermutationMatrix> Pl_blocked2;
    std::vector<PermutationMatrix> Pr_blocked2;
    std::vector<PermutationMatrix> Pl_blocked_total;
    std::vector<PermutationMatrix> Pr_blocked_total;
    MatRealAllocated A_copy(2*max_val, 2*max_val);
    MatRealAllocated A_verification(2*max_val, 2*max_val);
    MatRealAllocated A_verification_T(2*max_val, 2*max_val);
    MatRealAllocated L(2*max_val, 2*max_val);
    MatRealAllocated U(2*max_val, 2*max_val);
    MatRealAllocated B(2*max_val, 2*max_val);
    MatRealAllocated B_tilde(2*max_val, 2*max_val);

    A_full.reserve(nb_runs);
    Pl_full.reserve(nb_runs);
    Pr_full.reserve(nb_runs);
    A_blocked.reserve(nb_runs);
    Pl_blocked1.reserve(nb_runs);
    Pr_blocked1.reserve(nb_runs);
    Pl_rank.reserve(nb_runs);
    Pl_blocked2.reserve(nb_runs);
    Pr_blocked2.reserve(nb_runs);
    Pl_blocked_total.reserve(nb_runs);
    Pr_blocked_total.reserve(nb_runs);
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
                if (j >= nx[k] || i < nx[k]){ // transposed
                    A_blocked[k](i,j) = A_full[k](i,j);
                } else {
                    A_blocked[k](i,j) = 0.0; // zero out the top-left block
                    A_full[k](i,j) = 0.0;
                }
            }
        }
        Pl_full.emplace_back(m);
        Pr_full.emplace_back(n);
        Pl_blocked1.emplace_back(m);
        Pr_blocked1.emplace_back(n);
        Pl_rank.emplace_back(m);
        Pl_blocked2.emplace_back(m);
        Pr_blocked2.emplace_back(n);
        Pl_blocked_total.emplace_back(m);
        Pr_blocked_total.emplace_back(n);
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

        // store current matrix for verification later
        blasfeo_dgecp(nu[i]+nx[i], ng[i]+nx[i], &A_full[i].mat(), 0, 0, &A_copy.mat(), 0, 0);

        // perform lu
        auto start = std::chrono::high_resolution_clock::now();
        fatrop_lu_fact_transposed(ng[i]+nx[i], nu[i]+nx[i], nu[i]+nx[i], r, &A_full[i].mat(), Pl_full[i], Pr_full[i], 1e-5);
        auto end = std::chrono::high_resolution_clock::now();
        // time_full[i] = std::chrono::duration<double>(end - start).count();
        time_full[i] = std::chrono::duration<double, std::micro>(end - start).count();

        // verify
        if (verify && !verify_lu(A_full[i], A_copy, A_verification, A_verification_T, L, U, Pl_full[i], Pr_full[i], r, ng[i]+nx[i], nu[i]+nx[i])){
            std::cout << "Verification failed for full LU at run " << i+1 << std::endl;
            return -1;
        }
    }
    std::cout << "\nfull LU done" << std::endl;

    // perform blocked lu timings
    int r1 = 0; 
    int r2 = 0;
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        // std::cout << "running blocked LU ... " << std::setw(4) << std::setprecision(3) << progress << "%\r" << std::flush;
        
        blasfeo_dgese(A_copy.m(), A_copy.n(), 0, &A_copy.mat(), 0, 0);
        blasfeo_dgecp(nu[i]+nx[i], ng[i]+nx[i], &A_blocked[i].mat(), 0, 0, &A_copy.mat(), 0, 0);

        std::cout << "\nng = " << ng[i] << "\nnu = " << nu[i] << "\nnx = " << nx[i] << std::endl;
        write_np_matrix(A_blocked[i], nu[i]+nx[i], ng[i]+nx[i], "A_blocked");

        auto start = std::chrono::high_resolution_clock::now();
        // fatrop_lu_fact_transposed(nx[i], nx[i], nx[i], r1, &A_blocked[i].mat(), Pl_blocked1[i], Pr_blocked1[i], 1e-5);
        // fatrop_lu_fact_transposed(ng[i], nu[i], nu[i], r2, &A_blocked[i].mat(), nx[i], nx[i], Pl_blocked2[i], Pr_blocked2[i], 1e-5);
        // gecp(nx[i], ng[i], A_blocked[i], 0, nx[i], B, 0, 0);
        // Pr_blocked1[i].apply_inverse_on_rows(r1, &B.mat(), 0);
        // Pl_blocked2[i].apply_on_cols(r2, &B.mat());
        // blasfeo_dtrsm_llnn(nx[i], ng[i], 1.0, &A_blocked[i].mat(), 0, 0, &B.mat(), 0, 0, &B_tilde.mat(), 0, 0);

        // lu of top-left block
        fatrop_lu_fact_transposed(nx[i], nx[i], nx[i], r1, &A_blocked[i].mat(), Pl_blocked1[i], Pr_blocked1[i], 1e-5);
        std::cout << "first lu done (rank: " << r1 << ")" << std::endl;
        for (int k = 0; k < ng[k]; k++){ Pl_rank[i][r1 + k] = nx[k] + k;} // TODO: this is incorrect
        std::cout << "Pl_rank constructed" << std::endl;

        // permute rows of matrix
        std::cout << "A_blocked before permuting rows: \n" << A_blocked[i] << std::endl;
        Pl_rank[i].apply_inverse_on_cols(nx[i]+ng[i], &A_blocked[i].mat());
        std::cout << "permuted rows" << std::endl;
        std::cout << "A_blocked after permuting rows: \n" << A_blocked[i] << std::endl;

        // scaling bottom-left
        gecp(nx[i], ng[i], A_blocked[i], 0, r1, B, 0, 0);
        Pr_blocked1[i].apply_inverse_on_rows(r1, &B.mat(), 0);
        std::cout << "permuted rows of B" << std::endl;

        // compute K3 and K4
        blasfeo_dtrsm_llnn(r1, ng[i], 1.0, &A_blocked[i].mat(), 0, 0, &B.mat(), 0, 0, &A_blocked[i].mat(), 0, r1);
        std::cout << "computed K3" << std::endl;
        std::cout << "A_blocked after computing K3: \n" << A_blocked[i] << std::endl;

        blasfeo_dtrmm_lutn(nx[i]-r1, ng[i], -1.0, &A_blocked[i].mat(), 0, r1, &A_blocked[i].mat(), r1, 0, &B.mat(), 0, 0);
        blasfeo_dgead(nx[i]-r1, ng[i], 1.0, &B.mat(), 0, 0, &A_blocked[i].mat(), r1, r1);
        std::cout << "computed K4" << std::endl;
        std::cout << "A_blocked after computing K4: \n" << A_blocked[i] << std::endl;

        // second LU decomposition
        fatrop_lu_fact_transposed(ng[i], nx[i]-r1+nu[i], nx[i]-r1+nu[i], r2, &A_blocked[i].mat(), r1, r1, Pl_blocked2[i], Pr_blocked2[i], 1e-5);
        // Pl_blocked2[i].apply_on_cols(r2, &A_blocked[i].mat(), r1); // TODO: figure out how to do this permutation on K3
        std::cout << "second lu done" << std::endl;
        std::cout << "A_blocked after second lu: \n" << A_blocked[i] << std::endl;

        // construct Pl and Pr for the full matrix
        // Pl_blocked1[i].apply(r1, Pl_blocked_total[i], 0);
        // Pl_rank[i].apply(r1, Pl_blocked_total[i], 0);
        // Pl_blocked2[i].apply(r2, Pl_blocked_total[i], r1);

        auto end = std::chrono::high_resolution_clock::now();
        time_blocked[i] = std::chrono::duration<double, std::micro>(end - start).count();

        if (verify){
        bool check_block1 = verify_lu(A_blocked[i], A_copy, A_verification, A_verification_T, L, U, Pl_blocked1[i], Pr_blocked1[i], r1, nx[i], nx[i]);
        if (!check_block1){
            std::cout << "Verification failed for blocked LU (block 1) at run " << i+1 << std::endl;
            return -1;
        }
        bool check_block2 = verify_lu(A_blocked[i], A_copy, A_verification, A_verification_T, L, U, Pl_blocked2[i], Pr_blocked2[i], r2, ng[i], nu[i], nx[i], nx[i]);
        if (!check_block2){
            // std::cout << "\nng = " << ng[i] << "\nnu = " << nu[i] << "\nnx = " << nx[i] << std::endl;
            // write_np_matrix(A_blocked[i], nu[i]+nx[i], ng[i]+nx[i], "A_blocked");
            // std::cout << "Pl: \n" << Pl_blocked2[i] << std::endl;
            // std::cout << "Pr: \n" << Pr_blocked2[i] << std::endl;
            std::cout << "Verification failed for blocked LU (block 2) at run " << i+1 << std::endl;
            return -1;
        }
        // bool check_block_full = verify_blocked_lu(A_blocked[i], B_tilde, A_copy, 
        //                                           A_verification, A_verification_T, L, U, 
        //                                           Pl_blocked1[i], Pr_blocked1[i], r1, 
        //                                           Pl_blocked2[i], Pr_blocked2[i], r2, ng[i], nu[i], nx[i]);
        std::cout << "verifying full blocked LU ... " << std::endl;
        bool check_block_full = verify_blocked_lu_new(A_blocked[i], A_copy, 
                                                  A_verification, A_verification_T, L, U, 
                                                  Pl_blocked1[i], Pr_blocked1[i], r1,
                                                  Pl_rank[i],  
                                                  Pl_blocked2[i], Pr_blocked2[i], r2, ng[i], nu[i], nx[i]);
        if (!check_block_full){
            std::cout << "Verification failed for blocked LU (full) at run " << i+1 << std::endl;
            return -1;
        }
        }        
    }
    std::cout << "\nblocked LU done" << std::endl;

    if (!write_csv){
        return 0;
    }
    // write csv file
    std::ofstream csv_file(file_name);
    csv_file << "nu,nx,ng,time_full,time_blocked\n";
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        std::cout << "writing csv ... " << std::setw(4) << std::setprecision(3) << progress << "%\r" << std::flush;
        csv_file << nu[i] << "," << nx[i] << "," << ng[i] << "," << time_full[i] << "," << time_blocked[i] << "\n";
    }
    std::cout << "\ncsv writing done" << std::endl;

    std::cout << "average time for full LU:    " << std::accumulate(time_full.begin(), time_full.end(), 0.0) / nb_runs << " us" << std::endl;
    std::cout << "average time for blocked LU: " << std::accumulate(time_blocked.begin(), time_blocked.end(), 0.0) / nb_runs << " us" << std::endl;
}
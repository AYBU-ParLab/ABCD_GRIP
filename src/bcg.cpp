// Copyright Institut National Polytechnique de Toulouse (2014) 
// Contributor(s) :
// M. Zenadi <mzenadi@enseeiht.fr>
// D. Ruiz <ruiz@enseeiht.fr>
// R. Guivarch <guivarch@enseeiht.fr>

// This software is governed by the CeCILL-C license under French law and
// abiding by the rules of distribution of free software.  You can  use, 
// modify and/ or redistribute the software under the terms of the CeCILL-C
// license as circulated by CEA, CNRS and INRIA at the following URL
// "http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html"

// As a counterpart to the access to the source code and  rights to copy,
// modify and redistribute granted by the license, users are provided only
// with a limited warranty  and the software's author,  the holder of the
// economic rights,  and the successive licensors  have only  limited
// liability. 

// In this respect, the user's attention is drawn to the risks associated
// with loading,  using,  modifying and/or developing or reproducing the
// software by the user in light of its specific status of free software,
// that may mean  that it is complicated to manipulate,  and  that  also
// therefore means  that it is reserved for developers  and  experienced
// professionals having in-depth computer knowledge. Users are therefore
// encouraged to load and test the software's suitability as regards their
// requirements in conditions enabling the security of their systems and/or 
// data to be ensured and,  more generally, to use and operate it in the 
// same conditions as regards security. 

// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL-C license and that you accept its terms.

/*!
 * \file bcg.cpp
 * \brief Implementation of the stabilized block conjugate gradient algorithm
 * \author R. Guivarch, P. Leleux, D. Ruiz, S. Torun, M. Zenadi
 * \version 1.0
 */

#include <abcd.h>
#include "blas.h"

#include <iostream>



 void abcd::cg(MV_ColMat_double &b)
{
    std::streamsize oldprec = std::cout.precision();
    double t1_total, t2_total,t3_total =0 ,t4_total = 0;
    
    const double threshold = dcntl[Controls::threshold];
    const int block_size = icntl[Controls::block_size];    
    const int itmax = icntl[Controls::itmax];
    	
    if(!use_xk) {
        Xk = MV_ColMat_double(n, nrhs, 0);
    }

    MV_ColMat_double u(m, nrhs, 0);
    // get a reference to the nrhs first columns
    u = b(MV_VecIndex(0, b.dim(0)-1), MV_VecIndex(0,nrhs-1));

    MV_ColMat_double p(n, 1, 0);
    MV_ColMat_double qp(n, 1, 0);
    MV_ColMat_double r(n, 1, 0);


    MV_ColMat_double gu, bu, pl;
    
    double thresh = threshold;

    double *qp_ptr = qp.ptr();

    nrmB = std::vector<double>(nrhs, 0);
    VECTOR_double u_j = u(0);
    double lnrmBs = infNorm(u_j);
    mpi::all_reduce(inter_comm, &lnrmBs, 1,  &nrmB[0] , mpi::maximum<double>());

    VECTOR_double u_k = u(0);
     
    mpi::broadcast(inter_comm, nrmMtx, 0);

    // **************************************************
    // ITERATION k = 0                                 *
    // **************************************************

    t1_total = MPI_Wtime();
    if(use_xk) {
        MV_ColMat_double sp = sumProject(1e0, b, -1e0, Xk);
        r.setCols(sp, 0, 1);
        cout << "use_xk" << endl;
    } else {
        MV_ColMat_double sp = sumProject(1e0, b, 0, Xk);
        r.setCols(sp, 0, 1);         
    }
    
	//~ if(IRANK==7){ 
		//~ for(int k=0;k<column_index.size();k++) {
			//~ for(int i=0;i<column_index[k].size();i++) cout<< column_index[k][i] << " ";
			//~ cout << endl;
		//~ }
		//~ for(int k=0;k<local_column_index.size();k++) {
			//~ for(int i=0;i<local_column_index[k].size();i++) cout<< local_column_index[k][i] << " ";
			//~ cout << endl;
		//~ }
		//~ for(int k=0;k<loc_merge_index.size();k++) {
			//~ cout<< loc_merge_index[k] << " _ ";			
		//~ }
		//~ cout << endl;
		//cout << partitions[IRANK] << endl;
		//cout << endl << parts[IRANK] << endl;
		//~ ////~ cout << "\n rank: "<<IRANK<<" m: " << m << " r  \n" << r << endl;   
		//~ }
	
	//~ sleep(1);
	
	//~ sleep(1);
	//~ cout << "\n rank: "<<IRANK<<" m: " << m << " n: "<< n << " loc_n " << loc_n <<" merge size "<< merge_index.size()<< endl;   
	
	//~ if(IRANK==1){ for(int i=0;i<column_index[0].size();i++) cout<< column_index[0][i] << " ";
		//~ cout << "\n rank: "<<IRANK<<" m: " << m << " r  \n" << r << endl;   }
	//~ sleep(1);
	//~ if(IRANK==6){ for(int i=0;i<column_index[0].size();i++) cout<< column_index[0][i] << " ";
		//~ cout << "\n rank: "<<IRANK<<" m: " << m << " r  \n" << r << endl;   }
	//~ sleep(1);
	//~ if(IRANK==7){ for(int i=0;i<column_index[0].size();i++) cout<< column_index[0][i] << " ";
		//~ cout << "\n rank: "<<IRANK<<" m: " << m << " r  \n" << r << endl;   }
	//~ sleep(1);    
    //~ return;

	//~ if(IRANK==0)
		//~ {
			//~ for (int i =0 ; i< n ; i++)
			//~ {
				//~ if(comm_map[i] == 1) {
					//~ cout <<IRANK << ") parallel_cg " << parallel_cg<< " b size " << b.dim(0) << " " << i <<" " << r(i,0) << endl;
				//~ }
			//~ }
		//~ }
	
    t1_total = MPI_Wtime() - t1_total;

    p = r;
    
    
    VECTOR_double x = Xk(0);
    double *x_ptr = x.ptr();
    
    double * p_ptr = p.ptr();
    double * r_ptr = r.ptr();    
    
    VECTOR_double rnew(n, 0.0);   
        
    double *rnew_ptr = rnew.ptr();
    
    double scalar_gamma, scalar_alpha;
    int it = 0;
    double rho = 1;
    
    double ti = MPI_Wtime();

    t2_total = MPI_Wtime();
    rho = compute_rho(x, u_k);
    t2_total = MPI_Wtime() - t2_total;
    if(comm.rank() == 0) {
        LINFO2 << "ITERATION 0  rho = " << scientific << rho << setprecision(oldprec);
    }
    
    double scalar_r, scalar_r_new;    
    
    scalar_r = abcd::parallelddot(r_ptr, r_ptr);
    
    while(true) {
        it++;
        
        double t = MPI_Wtime();		
        // qp = Hp
        qp = sumProject(0e0, b, 1e0, p);        
        double t1 = MPI_Wtime() - t;
                
        double *qp_k_ptr = qp.ptr();
        
        scalar_gamma =  scalar_r / abcd::parallelddot(p_ptr, qp_k_ptr);             
                
                
        //~ abcd::scalarmult(p_k , scalar_gamma, tem_k);        
        //~ abcd::vectoradd(x , tem_k);        
        abcd::scalarmult_vectoradd(p_ptr , scalar_gamma ,x_ptr,x_ptr);
        
        //~ abcd::scalarmult(qp_k , scalar_gamma);        
        //~ abcd::vectorsubtract(r_k , qp_k, rnew_k);
        abcd::scalarmult_vectoradd(qp_k_ptr , -scalar_gamma ,r_ptr, rnew_ptr);
        
        scalar_r_new = abcd::parallelddot(rnew_ptr, rnew_ptr);        
        scalar_alpha =  scalar_r_new / scalar_r ; 
                
        scalar_r = scalar_r_new;
        
        //~ abcd::scalarmult(p_k , scalar_alpha, tem_k);        
        //~ abcd::vectoradd(rnew_k , tem_k ,p_k);        
        abcd::scalarmult_vectoradd(p_ptr , scalar_alpha ,rnew_ptr, p_ptr);
        
		double t2 = MPI_Wtime();
		//if(itmax < 0)
			rho = abcd::compute_rho(x, u_k);
        t2 = MPI_Wtime() - t2;
        
        t = MPI_Wtime() - t;
        if(comm.rank() == 0 && icntl[Controls::verbose_level] >= 2) {
            int ev = icntl[Controls::verbose_level] >= 4 ? 1 : 10;
            LOG_EVERY_N(ev, INFO) << "ITERATION " << it <<
                " rho = " << scientific << rho <<
                "  Timings: " << setprecision(2) << t <<
                setprecision(oldprec); // put precision back to what it was before
            
        }
                
        if((it >= itmax) ||	 (rho < thresh)) break; 
        
        for(int ii=0;ii<loc_merge_index.size();ii++)
			r_ptr[ii] = rnew_ptr[ii];
        //~ r_k = rnew_k;
         
        t1_total += t1;
        t2_total += t2;
    }
    
    //~ maxiter = it;
    
    //~ Xk(0) = x;
    //~ r(0) = r_k;

    if(inter_comm.rank() == 0) {
        LINFO << "CG Rho: " << scientific << rho ;
        LINFO << "CG Iterations : " << setprecision(2) << it ;
        LINFO << "CG TIME : " << scientific << MPI_Wtime() - ti ;
        LINFO << "SumProject time : " << t1_total ;
        LINFO << "Rho Computation time : " << t2_total ;
    }
    if (icntl[Controls::aug_type] != 0)
        return;

    info[Controls::nb_iter] = it;

    if(IRANK == 0) {
        solution = MV_ColMat_double(n_o, nrhs, 0);
        sol = solution.ptr();
    }
    
    centralizeVector(sol, n_o, nrhs, Xk.ptr(), n, nrhs, glob_to_local_ind, &dcol_[0]);
}


/*!
 *  \brief Uses Stabilized Block-CG to solve Hx = k
 *
 *  Uses Stabilized Block-CG to solve Hx = k where H is the sum of projectors
 *  and k is \sum A_i^+ b_i
 *  The block-size is defined in icntl[Controls::block_size]
 *
 *  \param b: the local right hand side
 *
 */
void abcd::bcg(MV_ColMat_double &b)
{
    std::streamsize oldprec = std::cout.precision();
    double t1_total, t2_total;

    // parameters of BCG
    const double threshold = dcntl[Controls::threshold];
    // s is the block size of the current run (may decrease when some vector converges)
    int s = std::max<int>(icntl[Controls::block_size], nrhs);
    const int itmax = icntl[Controls::itmax];
    // check max iteration
    if (itmax < 0) {
        info[Controls::status] = -11;
        mpi::broadcast(intra_comm, info[Controls::status], 0);

        throw std::runtime_error("Max iter number should be at least zero (0)");
    }

    // if no starting point used, init it with all zeros
    if(!use_xk) {
        Xk = MV_ColMat_double(n, nrhs, 0);
    // else init starting point for all block_size vectors
    } else if (Xk.dim(1) != s) {
        MV_ColMat_double Xk_tmp = MV_ColMat_double(n, s, 0);
        for (int iii=0; iii < s; ++iii) {
            Xk_tmp(MV_VecIndex(0, n-1), iii) = Xk;
        }
        Xk=Xk_tmp;
    }

    // get a reference to the nrhs first columns
    MV_ColMat_double u(m, nrhs, 0);
    u = b(MV_VecIndex(0, b.dim(0)-1), MV_VecIndex(0,nrhs-1));

    // initialize BCG vectors
    MV_ColMat_double p(n, s, 0); // direction
    MV_ColMat_double qp(n, s, 0); // ???
    double *qp_ptr = qp.ptr();
    MV_ColMat_double r(n, s, 0); // residual
    MV_ColMat_double gammak(s, s, 0); // gamma_k
    MV_ColMat_double betak(s, s, 0); // beta_k
    double *betak_ptr = betak.ptr();
    MV_ColMat_double lambdak(s, nrhs, 0); // lambda_k
    double *l_ptr = lambdak.ptr();
    MV_ColMat_double prod_gamma(nrhs, nrhs, 0); // product(k=j..0) of the gamma_k
    MV_ColMat_double e1(s, nrhs, 0); // ???
    MV_ColMat_double gu, bu, pl; // ???

    // Compute parallely Infinite norm of the RHS (max(bi))
    nrmB = std::vector<double>(nrhs, 0);
    for(int j = 0; j < nrhs; ++j) {
        VECTOR_double u_j = u(j);
        double lnrmBs = infNorm(u_j);
        // Sync B norm :
        mpi::all_reduce(inter_comm, &lnrmBs, 1,  &nrmB[0] + j, mpi::maximum<double>());
    }

    // Constants for the BLAS calls
    for(int k =0; k < e1.dim(1); k++) e1(0,k) = 1;
    char up = 'U';
    char left = 'L';
    char right = 'R';
    char tr = 'T';
    char notr = 'N';
    double alpha = 1;


    // **************************************************
    // ITERATION k = 0                                 *
    // **************************************************
    t1_total = MPI_Wtime();
    // Compute R(0) as sum of projections from b
    // if starting vector, R(0)=B+cst*HXk
    double cst=use_xk ? cst=-1e0 : 0.0; // if cst is 0, Xk won't be considered in sumProject
    MV_ColMat_double sp = sumProject(1e0, b, cst, Xk);
    r.setCols(sp, 0, s);

    t1_total = MPI_Wtime() - t1_total;

    // gamma_0, Rbar(0) = Stab(R(0))
    // Rbar(0) = R(0)*gamma_0^-1
#ifdef WIP
    if(icntl[Controls::use_gmgs2] != 0){
        gmgs2(r, r, gammak, s, false);
    } else
#endif //WIP
    if(gqr(r, r, gammak, s, false) != 0){
        gmgs2(r, r, gammak, s, false);
    }

    // product of gamma_k = gamma_0
    p = r;
    {
        MV_ColMat_double gu = gammak(MV_VecIndex(0, nrhs -1), MV_VecIndex(0, nrhs -1));
        prod_gamma = upperMat(gu);
    }

    int it = 0;
    double rho = 1;
    rhoVector = std::vector<double>();
    scaledResidualVector = std::vector<double>();

    double ti = MPI_Wtime();

    t2_total = MPI_Wtime();
    // compute the initial backward residual
    rho = compute_rho(Xk, u);

    t2_total = MPI_Wtime() - t2_total;
    if(comm.rank() == 0) {
        LINFO2 << "ITERATION 0  rho = " << scientific << rho << setprecision(oldprec);
    }

    // **************************************************
    // ITERATIONs                                      *
    // **************************************************
    while(true) {
        it++;
        double t = MPI_Wtime();

        // Compute sum of projections: qp = Hp
        qp = sumProject(0e0, b, 1e0, p);

        double t1 = MPI_Wtime() - t;

        // beta_k, Pbar(k), HPbar(k) = Stab(P(k), HP(k))
        // betak^T betak = chol(p^Tqp)
#ifdef WIP
        if(icntl[Controls::use_gmgs2] != 0){
            gmgs2(p, qp, betak, s, true);
        } else
#endif //WIP
        if(gqr(p, qp, betak, s, true) != 0){
            gmgs2(p, qp, betak, s, true);
        }

        // beta_kX=ones() => lambda_k = beta_k^-T
        lambdak = e1;
        dtrsm_(&left, &up, &tr, &notr, &s, &nrhs, &alpha, betak_ptr, &s, l_ptr, &s);

        // pl = Pbar * lambda_k * prod_gamma
        lambdak = gemmColMat(lambdak, prod_gamma);
        pl = gemmColMat(p, lambdak);

        // x = x + pl
        Xk(MV_VecIndex(0, Xk.dim(0) - 1), MV_VecIndex(0, nrhs -1)) += 
            pl(MV_VecIndex(0, pl.dim(0)-1), MV_VecIndex(0, nrhs - 1));

        double t2 = MPI_Wtime();
        // compute the backward residual
        rho = abcd::compute_rho(Xk, u);
        t2 = MPI_Wtime() - t2;
        rhoVector.push_back(rho);
        scaledResidualVector.push_back(dinfo[Controls::scaled_residual]);

        // Test convergence
        if((rho < threshold) || (it >= itmax)) break;

        // beta_kX = HPbar => X=HPbar.beta_k^{-1} and R = Rbar - HPbar * B^-T
        dtrsm_(&right, &up, &tr, &notr, &n, &s, &alpha, betak_ptr, &s, qp_ptr, &n);
        r = r - qp;

        // gamma_k, Rbar(k) = Stab(R(k))
        // Rbar(k) = R(k)*gamma_k^-1
#ifdef WIP
        if(icntl[Controls::use_gmgs2] != 0){
            gmgs2(r, r, gammak, s, false);
        } else
#endif //WIP
        if(gqr(r, r, gammak, s, false) != 0){
            gmgs2(r, r, gammak, s, false);
        }

        // alpha_k = beta_k*gamma_(k+1)^T
        gu = gammak(MV_VecIndex(0, nrhs -1), MV_VecIndex(0, nrhs -1));
        gu = upperMat(gu);
        gammak = upperMat(gammak);
        bu = upperMat(betak);
        betak = gemmColMat(bu, gammak, false, true);

        // P = Rbar + Pbar*alpha
        p = r + gemmColMat(p, betak);

        // update product(k=j..0) of gamma_k = prod_gamma(k-1) * gamma_k
        prod_gamma = gemmColMat(gu, prod_gamma);

        // Display info on iterations
        t = MPI_Wtime() - t;
        if(comm.rank() == 0 && icntl[Controls::verbose_level] >= 2) {
            int ev = icntl[Controls::verbose_level] >= 3 ? 1 : 10;
            LOG_EVERY_N(ev, INFO) << "ITERATION " << it <<
                " rho = " << scientific << rho <<
                "  Timings: " << setprecision(2) << t <<
                setprecision(oldprec); // put precision back to what it was before

        }
        t1_total += t1;
        t2_total += t2;
    }

    // Display info post BCG
    if(inter_comm.rank() == 0) {
        LINFO2 << "BCG Rho: " << scientific << rho ;
        LINFO2 << "BCG Iterations : " << setprecision(2) << it ;
        LINFO2 << "BCG TIME : " << MPI_Wtime() - ti ;
        LINFO2 << "SumProject time : " << t1_total ;
        LINFO2 << "Rho Computation time : " << t2_total ;
    }

    // if augmentation, stop here
    if (icntl[Controls::aug_type] != 0)
        return;

    info[Controls::nb_iter] = it;

    // Centralize solution on root
    if(IRANK == 0) {
        solution = MV_ColMat_double(n_o, nrhs, 0);
        sol = solution.ptr();
    }
    centralizeVector(sol, n_o, nrhs, Xk.ptr(), n, nrhs, glob_to_local_ind, &dcol_[0]);
}               /* -----  end of function abcd::bcg  ----- */

/*!
 *  \brief Compute backward error from b and current iterate Xk
 *
 *  Compute backward error from b and current iterate Xk
 *
 *  \param x: current iterate
 *  \param u: right hand side
 *
 */
double abcd::compute_rho(MV_ColMat_double &x, MV_ColMat_double &u)
{
    double rho = 999.999, minNrmR = 999.999;
    int min_j = 0;

    // Compute norm of X and R
    VECTOR_double nrmX(nrhs, 0);
    VECTOR_double nrmR(nrhs, 0);
    abcd::get_nrmres(x, u, nrmR, nrmX);

    double temp_rho = 0;
    for(int j = 0; j < nrhs; ++j) {
        temp_rho = nrmR(j) / (nrmMtx*nrmX(j) + nrmB[j]);
        rho = temp_rho < rho ? temp_rho : rho;

        if (minNrmR > nrmR(j)){
            minNrmR =  nrmR(j);
            min_j = j;
        }
    }
    dinfo[Controls::backward] = rho;

    // Also get norm of Inf-norm of R and scaled residual
    dinfo[Controls::residual] = minNrmR;
    dinfo[Controls::scaled_residual] = minNrmR/nrmB[min_j];
    return rho;
}               /* -----  end of function abcd::compute_rho  ----- */


double abcd::compute_rho(VECTOR_double &x, VECTOR_double &u)
{
    double rho = 999.999;
    
    double nrmX;
    double nrmR;

    abcd::get_nrmres(x, u, nrmR, nrmX);
    
    //~ if(IRANK==1) cout << "nrmR " << nrmR << " " << nrmX << endl;
    
    double temp_rho = 0;
  
	//temp_rho = nrmR / nrmB[0];
	temp_rho = nrmR / (nrmMtx*nrmX + nrmB[0]);
	rho = temp_rho < rho ? temp_rho : rho;	
	        
    dinfo[Controls::backward] = rho;
    dinfo[Controls::residual] = nrmR;
    dinfo[Controls::scaled_residual] = nrmR/nrmB[0];
    
    //~ cout << nrmR << " " << nrmB[0] << endl;
    return rho;
}

/*!
 *  \brief Apply Generalized Modified Gram-Schmidt squared
 *
 *  Apply Generalized Modified Gram-Schmidt squared to orthogonalize P and AP. The orthogonalization
 *  can be in term of A orthogonality in case of use_a. Calls the actual implementation
 *
 *  \param P: first matrix to orthogonalize
 *  \param AP: first matrix to orthogonalize (equal to HP)
 *  \param R: output
 *  \param s: size of the output
 *  \param use_a: A-orthogonality ?
 *
 */
void abcd::gmgs2(MV_ColMat_double &P, MV_ColMat_double &AP, MV_ColMat_double &R, int s, bool use_a)
{
    // initialize the SPD matrix G for the norm and scalar product
    int *gr = new int[AP.dim(0)];
    int *gc = new int[AP.dim(0) + 1];
    double *gv = new double[AP.dim(0)];

    for(int i = 0; i < AP.dim(0); i++){
        gr[i] = i;
        gc[i] = i;
        gv[i] = 1;
    }
    gc[AP.dim(0)] = AP.dim(0);

    // initialize the G matrix
    CompCol_Mat_double G(AP.dim(0), AP.dim(0), AP.dim(0), gv, gr, gc);

    // launch actual GMGS2
    abcd::gmgs2(P, AP, R, G, s, use_a);
    delete[] gr;
    delete[] gc;
    delete[] gv;
}               /* -----  end of function abcd::gmgs2  ----- */

/*!
 *  \brief Apply Generalized QR
 *
 *  Apply Generalized QR to orthogonalize P and AP. The orthogonalization can be in term
 *  of A orthogonality in case of use_a. Calls the actual implementation. At the end:
 *   - P=Pchol(P^TP)^{-1}
 *   - use_a: AP=APchol(P^TAP)^{-1}
 *
 *  \param P: first matrix to orthogonalize
 *  \param AP: first matrix to orthogonalize (equal to HP)
 *  \param R: output
 *  \param s: size of the output
 *  \param use_a: A-orthogonality ?
 *
 */
int abcd::gqr(MV_ColMat_double &P, MV_ColMat_double &AP, MV_ColMat_double &R, int s, bool use_a)
{
    // initialize the SPD matrix G for the norm and scalar product
    int *gr = new int[AP.dim(0)];
    int *gc = new int[AP.dim(0) + 1];
    double *gv = new double[AP.dim(0)];

    for(int i = 0; i < AP.dim(0); i++){
        gr[i] = i;
        gc[i] = i;
        gv[i] = 1;
    }
    gc[AP.dim(0)] = AP.dim(0);

    // initialize the G matrix
    CompCol_Mat_double G(AP.dim(0), AP.dim(0), AP.dim(0), gv, gr, gc);

    delete[] gr;
    delete[] gc;
    delete[] gv;

    // launch actual GQR
    return gqr(P, AP, R, G, s, use_a);
}               /* -----  end of function abcd::gqr  ----- */

/*!
 *  \brief Apply Generalized Modified Gram-Schmidt squared
 *
 *  Apply Generalized Modified Gram-Schmidt squared to orthogonalize P and AP. The orthogonalization
 *  can be in term of A orthogonality in case of use_a. Calls the actual implementation
 *
 *  \param p: first matrix to orthogonalize
 *  \param ap: first matrix to orthogonalize (equal to HP)
 *  \param r: output
 *  \param g: SPD matrix for the scalar product and norm
 *  \param s: size of the output
 *  \param use_a: A-orthogonality ?
 *
 */
void abcd::gmgs2(MV_ColMat_double &p, MV_ColMat_double &ap, MV_ColMat_double &r,
                CompCol_Mat_double g, int s, bool use_a)
{
    r = MV_ColMat_double(s, s, 0);

    // OK!! we have our R here, lets have some fun :)
    for(int k = 0; k < s; k++) {
        // Compute dot product diag(r)=p*ap^T=||p||_a (ddot)
        VECTOR_double p_k = p(k);
        VECTOR_double ap_k = ap(k);
        r(k, k) = abcd::ddot(p_k, ap_k);

        // Check values not too small in r (p ortho to ap)
        if(abs(r(k, k)) < abs(r(0,0))*1e-16) {
            LWARNING << "PROBLEM IN GMGS : FOUND AN EXT. SMALL ELMENT " <<  r(k, k) << " postion " << k;
            r(k, k) = 1;
            if (k > 0) r(k, k) = pow(r(k -1, k -1), 2);
        }
        // Check if r value negative, make it positive
        if(abs(r(k, k)) < 0) {
            LWARNING << "PROBLEM IN GMGS : FOUND A NEGATIVE ELMENT " << r(k, k) << " postion " << k;
            r(k, k) = abs(r(k, k));
        }
        // Check values not zero (p ortho to ap)
        if(r(k, k) == 0) {
            LWARNING << "PROBLEM IN GMGS : FOUND A ZERO ELMENT " <<  r(k, k) << " postion " << k;
            r(k, k) = 1;
        }

        // if all is fine, do an sqrt:
        r(k, k) = sqrt(r(k, k));

        // check r value not NaN
        if (r(k,k) != r(k,k)) {
            info[Controls::status] = -12;
            mpi::broadcast(intra_comm, info[Controls::status], 0);

            throw std::runtime_error("Error with GMGS2, stability issues.");
        }

        //p_k = p_k/||p_k||_a
        p_k = p_k / r(k, k);
        p.setCol(p_k, k);
        //ap_k = ap_k/||p_k||_a
        if(use_a){
            ap_k = ap_k / r(k, k);
            ap.setCol(ap_k, k);
        }

        //p_k = p_k - proj_p_1(p_k) - ... - proj_p_(k-1)(p_k)
        //ap_k = ap_k - proj_p_1(ap_k) - ... - proj_p_(k-1)(ap_k)
        for(int j = k + 1; j < s; j++){
            VECTOR_double p_j = p(j);
            VECTOR_double ap_j = ap(j);

            // proj_p_k: p_k*(p_k*ap_j)
            r(k, j) = abcd::ddot(p_k, ap_j);
            p_j = p_j - p_k * r(k, j);
            if(use_a) {
                ap_j = ap_j - ap_k * r(k, j);
            }
        }
    }
}               /* -----  end of function abcd::gmgs2  ----- */

/*!
 *  \brief Apply Generalized QR
 *
 *  Apply Generalized QR to orthogonalize P and AP. The orthogonalization can be in term of
 *  A orthogonality in case of use_a. At the end:
 *   - P=Pchol(P^TP)^{-1}
 *   - use_a: AP=APchol(P^TAP)^{-1}
 *
 *  \param p: first matrix to orthogonalize
 *  \param ap: first matrix to orthogonalize (equal to HP)
 *  \param r: output
 *  \param g: SPD matrix for the scalar product and norm
 *  \param s: size of the output
 *  \param use_a: A-orthogonality ?
 *
 */
int abcd::gqr(MV_ColMat_double &p, MV_ColMat_double &ap, MV_ColMat_double &r,
              CompCol_Mat_double g, int s, bool use_a)
{
    // initialize local matrices p, ap and r and pointers to their values
    MV_ColMat_double loc_p(n, s, 0);
    double *p_ptr = loc_p.ptr();
    MV_ColMat_double loc_ap(n, s, 0);
    double *ap_ptr = loc_ap.ptr();
    MV_ColMat_double loc_r(s, s, 0);
    double *l_r_ptr = loc_r.ptr();

    int pos = 0;
    // if use a, p/ap are initialized until #cols(A) else until #rows(p)
    int end = use_a ? n : p.dim(0);
    for(int i = 0; i < end; i++) {
        if(comm_map[i] == 1) {
            for(int j = 0; j < s; j++) {
                loc_p(pos, j) = p(i, j);
                if (use_a) loc_ap(pos, j) = ap(i, j);
            }
            pos++;
        }
    }

    // alpha, beta for sumProject
    double alpha, beta;
    alpha = 1;
    beta  = 0;

    // Constants for the BLAS calls
    int ierr = 0;
    char no = 'N';
    char trans = 'T';
    char up = 'U';
    char right = 'R';

    // Compute R = P'AP (use_a) or R=P'P with BLAS
    if(use_a){
        int lda_p = loc_p.lda();
        int lda_ap = loc_ap.lda();
        int loc_n = ap.dim(0);

        dgemm_(&trans, &no, &s, &s, &loc_n, &alpha, p_ptr, &lda_p, ap_ptr, &lda_ap, &beta, l_r_ptr, &s);

    } else{
        int lda_p = loc_p.lda();
        int loc_n = p.dim(0);

        dgemm_(&trans, &no, &s, &s, &loc_n, &alpha, p_ptr, &lda_p, p_ptr, &lda_p, &beta, l_r_ptr, &s);
    }

    // reduce R throughout masters
    double *r_ptr = r.ptr(); // pointer to R
    mpi::all_reduce(inter_comm, l_r_ptr, s * s,  r_ptr, std::plus<double>());

    // BLAS cholesky factorization of R
    // P = PR^-1  <=> P^T = R^-T P^T
    dpotrf_(&up, &s, r_ptr, &s, &ierr);

    // Check cholesky
    if(ierr != 0){
        stringstream err;
        LWARNING << "PROBLEM IN GQR " << ierr << " " << inter_comm.rank();
        LWARNING << "Switching to GMGS";

        return ierr;
    }

    p_ptr = p.ptr();
    ap_ptr = ap.ptr();

    // Solve RX=P to get PR^{-1}
    dtrsm_(&right, &up, &no, &no, &n, &s, &alpha, r_ptr, &s, p_ptr, &n);
    // Solve RX=AP to get APR^{-1}
    if(use_a){
        dtrsm_(&right, &up, &no, &no, &n, &s, &alpha, r_ptr, &s, ap_ptr, &n);
    }

    return 0;
}               /* -----  end of function abcd::gqr  ----- */

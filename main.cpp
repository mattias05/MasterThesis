//  Created by Mattias Franchetto on 01/12/16.
//  Copyright (c) 2016 Mattias Franchetto. All rights reserved.

#include <iostream>
#include <vector>
#include <dlib-18.18/dlib/matrix.h>
#include <dlib-18.18/dlib/optimization.h>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <random>
#include <string.h>
#include <ctime>
#include <fstream>
#include <sstream>      // for std::stringstream
#include <algorithm>
#include <iomanip>      // std::setprecision
#include <cstring>
#include <exception>    // std::exception

#include "functionObjects.h"

#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

typedef dlib::matrix<double,0,1> column_vector;

int main(int argc, char *argv[]){

try{

    std::stringstream convert2(argv[1]); // set up a stringstream variable named convert1, initialized with the input from argv[1]
    int dirNumber;
    convert2 >> dirNumber; 

    std::stringstream convert1(argv[2]); // set up a stringstream variable named convert1, initialized with the input from argv[1]
    int fileNumber;
    convert1 >> fileNumber; 

    // Length of time series:
    int T = 10000;

    // initialize 
    std::vector<double> r(T);
    std::vector<double> X_real(T);
    std::vector<double> V_real(T);
    std::vector<int> I_real(T);
    std::vector<double> J_real(T);
    column_vector rtn(T);
    column_vector true_values(8);
    column_vector garch_param(4);
    column_vector parameters(8);
    column_vector newGradient(8);
    column_vector true_stderr(8);
    dlib::matrix<double,8,8> H, D, V, I, I_rob, pseudoHess, trueHessian, trueInformationMatrix;
    column_vector rob_stderr(8);
    
	// measure execution time
	clock_t start1;
	double duration1;
	duration1 = 0;
	start1 = clock();

    random_device rd;
    uint32_t seed=time(NULL);

    // Distributions needed for the computations / algorithm:
    std::default_random_engine generator;
    generator.seed( seed ); // Seeded differently each time

    std::exponential_distribution<double> exponential(1.0);
    std::normal_distribution<double> normal(0.0,1.0);

    // Exact values:
    // double r_bar = .00028;
    // double vhat = .25/sqrt(250.0);
    // double alpha = .05;
    // double beta = .94; 
    // double K0 = .04; 
    // double X_bar = -5.0;
    // double eta = 3.0;
    // double a = .996;

    double K0Sim[] = {.01, .04, .1};
    double XbarSim[] = {-1.0, -4.0, -7.0};
    double etaSim[] = {1.0, 4.0, 7.0};
    double aSim[] = {.98, .996, .998};
    
    int sizeK0Sim = sizeof(K0Sim)/sizeof(*K0Sim);
    int sizeXbarSim = sizeof(XbarSim)/sizeof(*XbarSim);
    int sizeEtaSim = sizeof(etaSim)/sizeof(*etaSim);
    int sizeASim = sizeof(aSim)/sizeof(*aSim);
    int combSim = sizeK0Sim*sizeXbarSim*sizeEtaSim*sizeASim;
    
    dlib::matrix<double> Msim(8,combSim);
    int hSim = 0;
    double startComb[combSim+1];
    
    while (hSim < combSim)
    {
        for (int i=0; i<sizeK0Sim; i++)
        {
            for (int j=0; j<sizeXbarSim; j++)
            {
                for (int k=0; k<sizeEtaSim; k++)
                {
                    for (int z=0; z<sizeASim; z++)
                    {
                        Msim(0,hSim) = r_bar; 	// rbar
                        Msim(1,hSim) = vhat; 	// sigmabar
                        Msim(2,hSim) = alpha;	// alpha
                        Msim(3,hSim) = beta;	// beta
                        Msim(4,hSim) = K0Sim[i];
                        Msim(5,hSim) = XbarSim[j];
                        Msim(6,hSim) = etaSim[k];
                        Msim(7,hSim) = aSim[z];
                        hSim++;
                    }
                }
            }   
        }
    }
 
    true_values = colm(Msim,dirNumber);
    trueHessian = Loglikelihood_hessian(rtn, true_values);
    trueInformationMatrix = dlib::inv(trueHessian);
    for (long kk = 0; kk<8; kk++){
		true_stderr(kk) = sqrt(trueInformationMatrix(kk,kk));
	}
    double r_bar = true_values(0);
    double vhat = true_values(1);
    double alpha = true_values(2);
    double beta = true_values(3); 
    double K0 = true_values(4); 
    double X_bar = true_values(5);
    double eta = true_values(6);
    double a = true_values(7);
    
    //--------------------------------------------------------------------------//
    // Simulating the Jump Diffusion model:
    //--------------------------------------------------------------------------//

    // Initialisation:
    X_real[0] = X_bar;
    V_real[0] = vhat * vhat;
    bernoulli_distribution bernoulli(L(X_real[0]));
    I_real[0] = bernoulli(generator);
    J_real[0] = I_real[0] * K0 * exponential(generator);

    r[0] = r_bar + K0 * L(X_real[0]) + sqrt(V_real[0]) * normal(generator) - J_real[0] * I_real[0];
    
    for (int i=1; i<T; i++) {
        X_real[i] = (1-a) * X_bar + a * X_real[i-1] + eta * (r[i-1] - r_bar);
        V_real[i] = vhat * vhat * (1.0 - alpha - beta) + alpha * (r[i-1] - r_bar)*(r[i-1] - r_bar) + beta * V_real[i-1];
        bernoulli_distribution bernoulli( L(X_real[i]) );

        I_real[i] = bernoulli(generator);
        J_real[i] = I_real[i] * K0 * exponential(generator); // OBS

        r[i] = r_bar + K0 * L(X_real[i]) + sqrt(V_real[i]) * normal(generator) - J_real[i] * I_real[i];
    }

    // convert array into dlib::column_vector
	for (int i=0; i<T; i++) {
        rtn(i) = r[i];
    }

    /*
    //----------------------//
    // Or Read S&P500 data  //
    //----------------------//
    std::vector<double> Price;
    FILE * myFile;
    myFile = fopen("SP50019502014.csv", "r");
    if (myFile!=NULL) {
        double aux_r;
        while (fscanf(myFile, "%lf\n", &aux_r) == 1){
            Price.push_back (aux_r);
        }
        fclose (myFile);
    }
    else {
        std::cout << "Unable to open the file \n";
        return 0;
    }

    int n = Price.size();
    // Return vector
    std::vector<double> ret(n-1);
    column_vector rtn(n-1-250);

    // calculate returns
    for (int i=0; i<n-1-250; i++) {
        ret[i] = log(Price[i+1]/Price[i]);
        rtn(i) = ret[i];
    }
    */

    double start_rbar = dlib::mean(rtn);
    double start_sigma = dlib::variance(rtn);
    //----------------------------------------------//
    //												//
    // 				GARCH(1,1) Model 				//
    //												//
    //----------------------------------------------//

    //-------------------------------------------------------------------------------
    // create a search-grid on the parameter space to select the best starting point
    //-------------------------------------------------------------------------------

    double rbarVec[] = {start_rbar, 2e-04, 3e-04, -start_rbar, -2e-04, -3e-04};
    double sigmaVec[] = {3e-04, start_sigma, 7e-04};
    double alphaVec[] = {.01, .03 , .05, .08};
    double betaVec[] = {.4, .5, .6, .7, .8, .9};
    
    int sizeRbar = sizeof(rbarVec)/sizeof(*rbarVec);
    int sizeSigma = sizeof(sigmaVec)/sizeof(*sigmaVec);
    int sizeAlpha = sizeof(alphaVec)/sizeof(*alphaVec);
    int sizeBeta = sizeof(betaVec)/sizeof(*betaVec);
    int comb = sizeRbar*sizeSigma*sizeAlpha*sizeBeta;
    
    dlib::matrix<double> M(4,comb);
    int h = 0;
    double startll[comb+1];
    
    while (h < comb)
    {
        for (int i=0; i<sizeSigma; i++)
        {
            for (int j=0; j<sizeAlpha; j++)
            {
                for (int k=0; k<sizeBeta; k++)
                {
                    for (int z=0; z<sizeRbar; z++)
                    {
                        M(0,h) = rbarVec[z];
                        M(1,h) = sigmaVec[i];
                        M(2,h) = alphaVec[j];
                        M(3,h) = betaVec[k];
                        startll[h] = garchLogLikelihood(rtn, colm(M,h));
                        h++;
                    }
                }
            }   
        }
    }
    
    int bestStart = 0;
    double bestLik = startll[0]; 
    for (int i=1; i<comb; i++){
        if (startll[i] < bestLik){
            bestStart = i;
            bestLik = startll[i];
        }
    }
    
    // set as a starting point, the point in the search-grid with the smallest negative log-likelihood
    garch_param = colm(M,bestStart);

    find_min(dlib::lbfgs_search_strategy(30),
                dlib::objective_delta_stop_strategy(1e-13),
                	// dlib::gradient_norm_stop_strategy(1e-7).be_verbose(),
		                GarchNoData(rtn), 
		                    gradientGarchNoData(rtn), 
		                        garch_param, 
		                            -10.0);

    //--------------------------------------//	
   	// 		End of GARCH(1,1) estimation	//
   	//--------------------------------------//	
	
    //--------------------------------------//	
    //										//
    // 	 Estimate (kappa, Xbar, eta, a)		//
    //										//
    //--------------------------------------//

    //--------------------------------------------------//
    // Search-grid to select the best starting point	//
    //--------------------------------------------------//

    double kappaVec[] = {.01, .03, .05, .07}; // s&p500
    double XbarVec[] = {-4.5, -4.7, -4.9, -5.1, -5.3, -5.5};
    // double XbarVec[] = {-3.0, -3.5 -4.0, -4.5 -5.0, -5.5, -6.0}; // s&p500
    double etaVec[] = {2.5, 2.7, 3.0, 3.2, 3.5}; // s&p500
    double aVec[] = {.98, .983, .986, .99, .993, .998};
    
    int sizeKappa = sizeof(kappaVec)/sizeof(*kappaVec);
    int sizeXbar = sizeof(XbarVec)/sizeof(*XbarVec);
    int sizeEta = sizeof(etaVec)/sizeof(*etaVec);
    int sizeA = sizeof(aVec)/sizeof(*aVec);
    int comb1 = sizeKappa*sizeXbar*sizeEta*sizeA;
    
    dlib::matrix<double> M1(8,comb1);
    M1 = 0.0;
    dlib::matrix<double> M0(8,comb1);
    M0 = 0.0;
    dlib::matrix<double,8,8> hess1, D2, I2, V2;
    column_vector stderror(8);
    
    int h1 = 0;
    double startll1[comb1];
    column_vector finallk(comb1);
    column_vector temp_parameters(8);
    
    while (h1 < comb1)
    {
        for (int i=0; i<sizeKappa; i++)
        {
            for (int j=0; j<sizeXbar; j++)
            {
                for (int k=0; k<sizeEta; k++)
                {
                    for (int z=0; z<sizeA; z++)
                    {
                    	M1(0,h1) = garch_param(0); 	// rbar
                        M1(1,h1) = garch_param(1); 	// sigmabar
                        M1(2,h1) = garch_param(2);	// alpha
                        M1(3,h1) = garch_param(3);	// beta
                        M1(4,h1) = kappaVec[i];
                        M1(5,h1) = XbarVec[j];
                        M1(6,h1) = etaVec[k];
                        M1(7,h1) = aVec[z];
                        
                        dlib::set_colm(M0,h1) = colm(M1,h1);

                        // loglikelihood in the starting parameter
                        startll1[h1] = Loglikelihood(rtn, colm(M1,h1));
                        h1++;
                    }
                }
            }   
        }
    }
    
    int bestStart1 = 0;
    double bestLik1 = startll1[0]; 
    for (int i=1; i<comb1; i++){
        if (startll1[i] < bestLik1){
            bestStart1 = i;
            bestLik1 = startll1[i];
        }
    }
    
    parameters = colm(M1,bestStart1);
    // epsilon size for derivative approximations
    double epsilon = 1e-7;
   
    //----------------------------------------------------------------------------
	// maximize Loglikelihood fcn wrt theta and save the opt_theta
	//----------------------------------------------------------------------------
    
	find_min(dlib::newton_search_strategy(hessianCompleteModelNoData(rtn)),
    			dlib::objective_delta_stop_strategy(1e-10),
				// dlib::gradient_norm_stop_strategy(1e-4, 20).be_verbose(),
			    	completeModelNoData(rtn),
			     		gradientCompleteModelNoData(rtn),
			     			parameters,
			     				-100.0);

    // update the gradient
	newGradient = Loglikelihood_derivative(rtn, parameters);

	//------------------------------------------//
    // 	check hessian positive-definite.	  	//
    // 	maximum or saddle point ?				//
    //------------------------------------------//

    // Compute the hessian in the optimal point we just found
	H = Loglikelihood_hessian(rtn, parameters);
	// Information matrix
	I = dlib::inv(H);
	
    dlib::eigenvalue_decomposition<dlib::matrix<double> > eig(H);

	V = eig.get_pseudo_v();
	D = eig.get_pseudo_d();

	for (long kk = 0; kk<8; kk++){
    	stderror(kk) = sqrt(I(kk,kk));
    }
    // divide by the sqrt(number of observation)
    stderror /= sqrt(rtn.size());
    
   	//--------------------------------------------------------------------------------------//
   	// Line-search in all eigenvectors' directions trying to minimize the norm(gradient)	//
   	//--------------------------------------------------------------------------------------//
    int maxdir = 0;
    for (int jj = 0; jj<8; jj++){

        int kk = 0;
        while (fabs(trans(newGradient) * colm(V,jj)) > 1e-4 && kk < 2){
        
            maxdir = jj;

           	column_vector direction(colm(V,maxdir));
           	column_vector starting_x(1);
           	starting_x = .0;
   	
           	find_min(dlib::newton_search_strategy(lineSearch_secondderivative(rtn, direction, parameters)),
            			// dlib::objective_delta_stop_strategy(1e-10).be_verbose(),
        				dlib::gradient_norm_stop_strategy(1e-7, 5),
        			    	lineSearch(rtn, direction, parameters),
        			     		lineSearch_derivative(rtn, direction, parameters),
        			     			starting_x,
        			     				-100.0);

           	newGradient = Loglikelihood_derivative(rtn, parameters + starting_x(0) * direction);

        	parameters = parameters + starting_x(0) * direction;

            // Compute the hessian in the optimal point we just found
            H = Loglikelihood_hessian(rtn, parameters);
        	// Information matrix
        	I = dlib::inv(H);

            // create an object with eigenvalue decomposition
        	dlib::eigenvalue_decomposition<dlib::matrix<double> > eig11(H);

            // get the eigenvectors
        	V = eig11.get_pseudo_v();
            // get the eigenvalues
        	D = eig11.get_pseudo_d();

            // standard errors of the estimates
        	for (long kk = 0; kk<8; kk++){
            	stderror(kk) = sqrt(I(kk,kk));
            }
            stderror /= sqrt(rtn.size());
            
            kk++;
        }
    }   

	// Estimating the computation time:
	duration1 += (clock() - start1) / (double) CLOCKS_PER_SEC;
	cout << duration1 << " seconds" << endl;

    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Solution:\n" << trans(parameters) << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "gradient value:\n" << trans(newGradient) << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Standard Errors:\n" << trans(stderror) << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;

    // save the data
    char filename[30];
    snprintf(filename, sizeof(filename), "data/case%d/simdata%d.csv", dirNumber, fileNumber);
    ofstream myfile_sim;
    myfile_sim.open (filename);
    for (int i=0; i<T; i++) {
        myfile_sim << X_real[i] << "," << V_real[i] <<  "," << I_real[i] << "," << J_real[i] << "," << r[i] << "\n";
    }
    myfile_sim.close();

    char filename1[30];
    snprintf(filename1, sizeof(filename1), "data/case%d/paramEst%d.csv", dirNumber, fileNumber);
    ofstream myfile_est;
    myfile_est.open (filename1);
    for (int i=0; i<8; i++) {
        myfile_est << parameters(i) << "," << newGradient(i) <<  "," << stderror(i) << "," << true_mean(i) << "," << true_stderr(i) "\n";
    }
    myfile_est.close();

	}
	catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}




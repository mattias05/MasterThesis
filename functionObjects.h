#include <iostream>
#include <vector>
#include </usr/local/include/dlib-18.18/dlib/matrix.h>
#include </usr/local/include/dlib-18.18/dlib/optimization.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <exception>      // std::exception

#include "modelFunctions.h"

#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

using namespace std;
using namespace dlib;

typedef matrix<double,0,1> column_vector;

//------------------------------------------------------------------------------------------------//
// This object is a "function model" which can be used with the find_min_trust_region() routine   //
//------------------------------------------------------------------------------------------------//
class loglikelihood_model
{
public:

    typedef ::column_vector column_vector;
    typedef matrix<double> general_matrix;

    loglikelihood_model (const column_vector& r)
    {
        data = r;
    }

    double operator() (const column_vector& param) const 
    {
        double ll = Loglikelihood(data, param);
        return ll; 
    }

    void get_derivative_and_hessian (const column_vector& param, column_vector& der, general_matrix& hess) const
    {
        der = Loglikelihood_derivative(data, param);
        hess = Loglikelihood_hessian(data, param);
    }

private:
    column_vector data;
};

//--------------------------------------------------------------------------------------------------------------------------//
// function-objects created to be able to minimize functions and use analytical gradients without reading data everytime.   //
// problem faced since DLIB optimization library requires objective function to receive in input only parameter vector      //
//--------------------------------------------------------------------------------------------------------------------------//

// function-object created to compute garch(1,1) log-likelihood
class GarchNoData{
public:

    GarchNoData (const column_vector& r)
    {
        data = r;
    }

    double operator() ( const column_vector& param) const
    {
        // return the log-likelihood of a garch(1,1) process with log-rtn = data and parameters = param
        double ll = garchLogLikelihood(data, param);
        return ll;
    }

private:
    column_vector data;
};

// function-object created to compute the log-likelihood of the complete model
class completeModelNoData{
public:

    completeModelNoData (const column_vector& r)
    {
        data = r;
    }

    double operator() ( const column_vector& param) const
    {
        // return the log-likelihood of a garch(1,1) process with log-rtn = data and parameters = param
        double ll = Loglikelihood(data, param);
        return ll;
    }

private:
    column_vector data;
};

// function-object created to freeze the first 4 parameters, already optimised with GARCH (sigma_bar, alpha, beta, rbar)
// and optimise only the remaining last 4 (kappa, Xbar, eta, a)
class completeModelNoGarchParams{
public:

    completeModelNoGarchParams (const column_vector& r, const column_vector& garchParam)
    {
        data = r;
        garchParameters = garchParam;
    }

    double operator() ( const column_vector& lastParam) const
    {
        // return the log-likelihood complete model
        column_vector param(8);
        param(0) = garchParameters(0);
        param(1) = garchParameters(1);
        param(2) = garchParameters(2);
        param(3) = garchParameters(3);
        param(4) = lastParam(0);
        param(5) = lastParam(1);
        param(6) = lastParam(2);
        param(7) = lastParam(3);
		        
        double ll = Loglikelihood(data, param);
        return ll;
    }

private:
    column_vector data;
    column_vector garchParameters;
};

// function-object created to compute the log-likelihood of the model with constant lambda, that is (a,eta) = 0,
// but providing anyway the 8-dimensional parameter vector
class completeModel6Params{
public:

    completeModel6Params (const column_vector& r)
    {
        data = r;
    }

    double operator() ( const column_vector& sixParam) const
    {
        // return the log-likelihood complete model
        column_vector param(8);
        param(0) = sixParam(0);
        param(1) = sixParam(1);
        param(2) = sixParam(2);
        param(3) = sixParam(3);
        param(4) = sixParam(4);
        param(5) = sixParam(5);
        param(6) = 0.0;
        param(7) = 0.0;
                
        double ll = Loglikelihood(data, param);
        return ll;
    }

private:
    column_vector data;
};

// function-object created to compute the log-likelihood of the model where one of the parameter is frozen
// pos = position of the parameter I would like to freeze
// val = value of the frozen parameter
class completeModel7Params{
public:

    completeModel7Params (const column_vector& r, const double val, const long pos)
    {
        data = r;
        val1 = val;
        pos1 = pos;
    }

    double operator() ( const column_vector& sevenParam) const
    {
        // return the log-likelihood complete model
        column_vector param(8);
        long j=0;
        for (long i=0; i<8; i++){
            if (i==pos1)
                param(i) = val1;
            else{
                param(i) = sevenParam(j);
                j++;
            }
        }       
        double ll = Loglikelihood(data, param);
        return ll;
    }

private:
    column_vector data;
    long pos1;
    double val1;
};

// function-object created to compute the gradient of the complete log-likelihood 
class gradientCompleteModelNoData{
public:

    gradientCompleteModelNoData (const column_vector& r)
    {
        data = r;
    }

    column_vector operator() ( const column_vector& param) const
    {
        // return the gradient of the log-likelihood of a garch(1,1) process with log-rtn = data and parameters = param
        column_vector grad = Loglikelihood_derivative(data, param);
        return grad;
    }

private:
    column_vector data;
};

// function-object created to freeze the first 4 parameters, already optimised with GARCH (sigma_bar, alpha, beta, rbar)
// and optimise only the remaining last 4 (kappa, Xbar, eta, a)
class gradientCompleteModelNoGarchParams{
public:

    gradientCompleteModelNoGarchParams (const column_vector& r, const column_vector& garchParam)
    {
        data = r;
        garchParameters = garchParam;
    }

    column_vector operator() ( const column_vector& lastParam) const
    {
        // return the log-likelihood complete model
        column_vector param(8);
        param(0) = garchParameters(0);
        param(1) = garchParameters(1);
        param(2) = garchParameters(2);
        param(3) = garchParameters(3);
        param(4) = lastParam(0);
        param(5) = lastParam(1);
        param(6) = lastParam(2);
        param(7) = lastParam(3);

        column_vector grad = Loglikelihood_derivative(data, param);
		column_vector smallGrad(4);
		smallGrad(0) = grad(4);
		smallGrad(1) = grad(5);
		smallGrad(2) = grad(6);
		smallGrad(3) = grad(7);
		
        return smallGrad;
    }

private:
    column_vector data;
    column_vector garchParameters;
};

// function-object created to compute the gradient of the log-likelihood where one of the parameter is frozen
// pos = position of the parameter I would like to freeze
// val = value of the frozen parameter
class gradientCompleteModel7Params{
public:

    gradientCompleteModel7Params (const column_vector& r, const double val, const long pos)
    {
        data = r;
        val1 = val;
        pos1 = pos;
    }

    column_vector operator() ( const column_vector& sevenParam) const
    {
        // return the 7-dimensional gradient
        column_vector param(8);
        long j=0;
        for (long i=0; i<8; i++){
            if (i==pos1)
                param(i) = val1;
            else{
                param(i) = sevenParam(j);
                j++;
            }
        }       
        column_vector grad = Loglikelihood_derivative(data, param);
        column_vector smallGrad(7);

        long k = 0;
        for (long i=0; i<8; i++){
            if (i != pos1){
                smallGrad(k) = grad(i);
                k++;
            }
        }      
        
        return smallGrad;
    }

private:
    column_vector data;
    long pos1;
    double val1;
};

// function-object created to compute the hessian of the log-likelihood of the complete model
class hessianCompleteModelNoData{
public:

    hessianCompleteModelNoData (const column_vector& r)
    {
        data = r;
    }

    dlib::matrix<double> operator() ( const column_vector& param) const
    {
        // return the gradient of the log-likelihood of a garch(1,1) process with log-rtn = data and parameters = param
        dlib::matrix<double> hess = Loglikelihood_hessian(data, param);
        return hess;
    }

private:
    column_vector data;
};

// function-object created to compute the gradient of the garch(1,1) log-likelihood
class gradientGarchNoData{
public:

    gradientGarchNoData (const column_vector& r)
    {
        data = r;
    }

    column_vector operator() ( const column_vector& param) const
    {
        // return the gradient of the log-likelihood of a garch(1,1) process with log-rtn = data and parameters = param
        column_vector grad = garchll_derivative(data, param);
        return grad;
    }

private:
    column_vector data;
};

// function-object created to compute the hessian of the log-likelihood where one of the parameter is frozen
// pos = position of the parameter I would like to freeze
// val = value of the frozen parameter
class hessianCompleteModel7Params{
public:

    hessianCompleteModel7Params (const column_vector& r, const double val, const long pos)
    {
        data = r;
        val1 = val;
        pos1 = pos;
    }

    dlib::matrix<double> operator() ( const column_vector& sevenParam) const
    {
        // return the 7x7 hessian
        column_vector param(8);
        long j=0;
        for (long i=0; i<8; i++){
            if (i==pos1)
                param(i) = val1;
            else{
                param(i) = sevenParam(j);
                j++;
            }
        }       
        
        dlib::matrix<double> hess = Loglikelihood_hessian(data, param);
        dlib::matrix<double,7,7> hessSmall;

        int k = 0;
        int l;
        for (int i=0; i<8; i++){
            if (i != pos1){
	            l = 0;
	            for (int j=0; j<8; j++){
	                if (j != pos1){
	                    hessSmall(k,l) = hess(i,j);
	                    l++;
	                }
	            }
	        k++;
	        }
        }      
        
        return hessSmall;
    }

private:
    column_vector data;
    long pos1;
    double val1;
};


// function-object created to perform line-search in a specified direction 
// in our case it will be the eigenvector of the parameter with the highest eigenvalue.
// param = parameter vector (where we are)
// V = direction
// data = log-returns
class lineSearch{
public:

    lineSearch (const column_vector& r, const column_vector& V_, const column_vector& param_)
    {
        data = r;
        V = V_;
        param = param_;
    }

    double operator() (const column_vector& x) const
    {
        double ll = Loglikelihood(data, param + x * V);
        return ll;
    }

private:
    column_vector data, V, param;
};

// function-object created to calculate the gradient in order to perform line-search in a specified direction 
// in our case it will be the eigenvector of the parameter with the highest eigenvalue.
// param = parameter vector (where we are)
// V = direction
// data = log-returns
class lineSearch_derivative{
public:

    lineSearch_derivative (const column_vector& r, const column_vector& V_, const column_vector& param_)
    {
        data = r;
        V = V_;
        param = param_;
    }

    column_vector operator() (const column_vector& x) const
    {
        column_vector grad;
        grad = Loglikelihood_derivative(data, param + x * V);

        return trans(grad) * V ;
    }

private:
    column_vector data, V, param;
};

// function-object created to calculate the hessian in order to perform line-search in a specified direction 
// in our case it will be the eigenvector of the parameter with the highest eigenvalue.
// param = parameter vector (where we are)
// V = direction
// data = log-returns
class lineSearch_secondderivative{
public:

    lineSearch_secondderivative (const column_vector& r, const column_vector& V_, const column_vector& param_)
    {
        data = r;
        V = V_;
        param = param_;
    }

    dlib::matrix<double> operator() (const column_vector& x) const
    {
        dlib::matrix<double> hess = Loglikelihood_hessian(data, param + x * V);
        
        return trans(V) * hess * V;
    }

private:
    column_vector data, V, param;
};




//
// Header file containing all the functions used in ThesisNoDataNoFunction.cpp file
//

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

#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

using namespace std;
using namespace dlib;

typedef matrix<double,0,1> column_vector;

double L(double x) {
  return (1.0L/(1.0L+exp(-1.0L*x)));
}

// Function used for sorting in absolute value
bool fun(double i, double j)
{
   return fabsl(i)>fabsl(j);
}

// Normal cumulative density function:
double pnorm(double q, double mu=0.0L, double sd=1.0L) {

	double SQRTHL = 7.071067811865475244008e-1L;
	double x, y, z;

	x = (q-mu)/sd * SQRTHL;
	z = abs(x);

	if( z < SQRTHL )
		y = 0.5L + 0.5L * erf(x);
	else{
		y = 0.5L * erfc(z);

		if( x > 0.0L )
		y = 1.0L - y;
	}
	return y;
	}

// sign function
double sign(double x) {
	if (x >= 0.0L) {
		return 1.0L;
	}
	else {
		return -1.0L;
	}
}

/*************************************************/
/*												 */
/*         		Auxilary function Rnorm		     */
/*	 Based on : G. Marsaglia (2004). Evaluating	 */
/*  	the Normal Distribution. Journal of 	 */
/*			Statistical Software 11(4).			 */
/*												 */
/*	Returns R(x)=[1-Phi(x)]/phi(x) if x>=0		 */
/* 	Returns R(x)=sqrt(2PI)*Phi(|x|)  if x<0		 */
/*												 */
/*************************************************/
double Rnorm(double x)
{
	long double R[30]=
	{1.2533141373155002512078826e+00L,
	4.2136922928805447322493433e-01L,
	2.3665238291356067062398594e-01L,
	1.6237766089686746181568210e-01L,
	1.2313196325793229628218074e-01L,
	9.9028596471731921395337189e-02L,
	8.2766286501369177252265050e-02L,
	7.1069580538852107090596842e-02L,
	6.2258665995026195776685945e-02L,
	5.5385651470100734467246739e-02L,
	4.9875925981836783658240561e-02L,
	4.5361207289993100875851688e-02L,
	4.1594702232575505419657616e-02L,
	3.8404893342102127679827361e-02L,
	3.5668904990075763502562778e-02L,
	3.3296419072497213381868402e-02L,
	3.1219571395243034937756812e-02L,
	2.9386387746754350419683055e-02L,
	2.7756393731398025502402434e-02L,
	2.6297602974252964377584116e-02L,
	2.4984404205720571147388395e-02L,
	2.3796049237106978988931619e-02L,
	2.2715551577751081925488961e-02L,
	2.1728871275070839467815251e-02L,
	2.0824302836246593878086164e-02L,
	1.9992009580853567311156686e-02L,
	1.9223665135847312181045008e-02L,
	1.8512174383012843787926668e-02L,
	1.7851454053792390770631005e-02L,
	1.7236258612862016439065156e-02L};
	
	long double SQRT2PI=2.50662827463100050241576528L; // sqrt(2*pi)
	
	int i,j=.5*(fabs(x)+1);
	
	if (j>11)
	{	
		long double h=-1.0L/(x*x), s=1.0L;
		for(i=100;i>-1;i--)
			s=fmal((2*i+1)*h,s,1.0L);
			
		if (x>=0.0L)
			return (double) s/x;
		else
			return (double) fmal(-s,expl(-.5L*x*x),SQRT2PI);
	}
	else
	{
		long double pwr=1.0L, a=R[j],z=2.0L*j, b=fma(a, z ,-1.0L), h=fma(-2.0L, j, fabsl(x)), s=fma(h, b, a),t=a,q=h*h;
		for(i=2;s!=t;i+=2){a=fma(z, b ,a)/i; b=fma(z, a, b)/(i+1); pwr*=q; s=fma(pwr,fma(h, b, a),(t=s));}
		
		if(x>=0)
			return (double) s;
		else
			return (double) fma(-s,expl(-.5L*x*x),SQRT2PI);
	}
}

// Normal density:
double dnorm(double x, double mu, double sd2) {
	// sd2 denotes the variance
	return exp(-.5L*(x-mu)*(x-mu)/sd2-.91893853320467274178L)/sqrt(sd2);
}

// Normal pdf used in the LogLikelihood fcn (Malevergne)
double normpdf(double x)
{
	return expl(-.5L*x*x);
}

// Normal pdf used in the LogLikelihood Gradient and Hessian (Malevergne)
double phi(double x)
{
	if (x>0.0)
		return 1.0L;
	return expl(-.5L*x*x-.91893853320467274178L);
}

// Exponential modified Gaussian density:
double demg(double x, double mu, double sd, double k) {
	return 1.0L/abs(k)*exp((x-mu)/k + sd*sd/(2.0L*k*k)) * 
		(1.0L - pnorm((sign(k)*((x-mu)/sd + sd/k))));
}

double Loglikelihood(const dlib::matrix<double,0,1>& r, const dlib::matrix<double,0,1>& param) {
		
	const long double mu=param(0);
	const long double sigma0=param(1);
	const long double alpha=param(2);
	const long double beta=param(3);
	const long double K0=param(4);
	const long double Xbar=param(5);
	const long double eta=param(6);
	const long double a=param(7);

	//----------------------------------------
	// Check constraints
	//----------------------------------------

	if (sigma0<0.0L){
		// std::cout << "Parameter SIGMA0 must be positive " << sigma0 << std::endl; 
		return 1e100;
    }
	if (alpha>1.0L || alpha<0.0L){
		// std::cout << "Parameter ALPHA must belong to (0,1) " << alpha << std::endl; 
		return 1e100;
	}
	
	if (beta>1.0L || beta<0.0L){
		// std::cout << "Parameter BETA must belong to (0,1) " << beta << std::endl; 
		return 1e100;
	}
	
	if (alpha+beta>1.0L){
		// std::cout << "ALPHA + BETA must be less than one " << alpha + beta << std::endl; 
		return 1e100;
	}
		
	if (eta<0.0L){
		// std::cout << "Parameter ETA must be positive" << std::endl; 
		return 1e100;
	}

	if (K0<0.0L){
		// std::cout << "Parameter KAPPA must be positive" << std::endl; 
		return 1e100;
	}
		
	if (a>1.0L || a<0.0L){
		// 	std::cout << "Parameter a must belong to (0,1) " << a << std::endl;
		return 1e100;
	}

	long T = r.size();
	
	// long time variance
	const long double omega = sigma0*sigma0*(1.0L-alpha-beta);

	// auxiliary variables for for rounding errors
	long double sum = 0.0L, c = 0.0L, y = 0.0L, tt=0.0L;
	long double s = 0.0L, cc = 0.0L, cs = 0.0L, ccs = 0.0L;
		
	//----------------------------------------
	// Step 0
	//----------------------------------------
	long double sigma2 = sigma0 * sigma0 * (1.0L - alpha);
	long double sigma = sqrtl(sigma2);
	long double X = Xbar;
	long double lambda = L(X);
	long double K = K0;
	long double mu_t = fmal(K, lambda, mu);

	long double A = (r(0) - mu_t)/sigma + sigma/K;
	long double l_aux = 0.0L;
	
	if (K == 0.0L)
		l_aux = 0.5L * (logl(sigma2) + (r(0)-mu_t) * (r(0)-mu_t)/sigma2);

	else if ((sign(K)*A) >= 0.0L){
		l_aux = lambda * fmal(sigma/fabsl(K), Rnorm(sign(K)*A), -1.0L);
		l_aux = -log1p(l_aux);
		l_aux+= 0.5L * (logl(sigma2) + (r(0)-mu_t) * (r(0)-mu_t)/sigma2);
	}	
	
	else {
		l_aux = lambda/fabsl(K) * Rnorm(sign(K)*A) + (1.0L-lambda)/sigma * normpdf(A);
		l_aux = -logl(l_aux);
		l_aux-= fmal(sigma2, 0.5L/K/K, (r(0)-mu_t)/K);
	}
	
	l_aux/=T;

	//----------------------------------------
	// Part to correct the errors
	//----------------------------------------

	tt = s + l_aux;
	if (fabsl(s) >= fabsl(l_aux))
		c = (s-tt) + l_aux;
	else
		c = (l_aux-tt) + s;
	
	s = tt;
	tt = cs + c;
	if (fabsl(cs) >= fabsl(c))
		cc = (cs-tt) + c;
	else
		cc = (c-tt) + cs;
		
	cs = tt;
	ccs = ccs + cc;

	//----------------------------------------
	// Step 1 to T-1
	//----------------------------------------
	for (long t=1; t<T; t++)
	{
		sigma2 *= beta;
		sigma2 += omega + alpha * (r(t-1)-mu) * (r(t-1)-mu);
		sigma = sqrtl(sigma2);
		X *= a;
		X += (1.0L-a) * Xbar + eta * (r(t-1)-mu);
		lambda = L(X);
		K = K0;
		mu_t = fmal(K, lambda, mu);

		A = (r(t)-mu_t)/sigma + sigma/K;
		
		if (K == 0.0L){
			l_aux = 0.5L * (logl(sigma2) + (r(t)-mu_t)*(r(t)-mu_t)/sigma2);
		}	
		
		else if ((sign(K)*A) >= 0.0L){

			l_aux = lambda * fmal(sigma/fabsl(K), Rnorm(A), -1.0L);
			l_aux = -log1p(l_aux);
			l_aux+= 0.5L * (logl(sigma2) + (r(t)-mu_t)*(r(t)-mu_t)/sigma2);
		}

		else {

			l_aux = lambda / fabsl(K) * Rnorm(sign(K)*A) + (1.0L-lambda)/sigma*normpdf(A);
			l_aux = -log(l_aux);
			l_aux-= fmal(sigma2, .5L/K/K, (r(t)-mu_t)/K);
		}
				
		l_aux/=T;

		//----------------------------------------
		// Part to correct the errors
		//----------------------------------------
		tt = s + l_aux;
		if (fabsl(s) >= fabsl(l_aux))
			c = (s-tt) + l_aux;
		else
			c = (l_aux-tt) + s;
		
		s = tt;
		tt = cs + c;
		if (fabsl(cs) >= fabsl(c))
			cc = (cs-tt) + c;
		else
			cc = (c-tt) + cs;
		
		cs = tt;
		ccs = ccs + cc;
	}
	
	return (double) ((s+cs+ccs)+0.91893853320467274178L);
}

const dlib::matrix<double,0,1> Loglikelihood_derivative(const dlib::matrix<double,0,1>& r, const dlib::matrix<double,0,1>& param){

	const double mu = param(0);
	const double sigma0 = param(1);
	const double alpha = param(2);
	const double beta = param(3);
	const double K0 = param(4);
	const double Xbar = param(5);
	const double eta = param(6);
	const double a = param(7);

	//----------------------------------------
	// Check constraints
	//----------------------------------------
	if (sigma0<0.0){
		// std::cout << "Parameter SIGMA0 must be positive" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}
		
	if (alpha>1.0 || alpha<0.0){
		// std::cout << "Parameter ALPHA must belong to (0,1)" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}
	
	if (beta>1.0 || beta<0.0){
		// std::cout << "Parameter BETA must belong to (0,1)" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}
	
	if (alpha+beta>0.999){
		// std::cout << "ALPHA + BETA must be less than one" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}
		
	if (eta<0.0){
		// std::cout << "Parameter ETA must be positive" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}

	if (K0<0.0){
		// std::cout << "Parameter KAPPA must be positive" << std::endl; 
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}
		
	if (a>1.0 || a<0.0){
		// std::cout << "Parameter a must belong to (0,1)" << std::endl;
		return dlib::derivative(Loglikelihood, 1e-7)(r, param);
	}

	double omega = sigma0 * sigma0 * (1.0 - alpha - beta);

	int T = r.size();
	
	//--------------------------------------//
	//										//
	// 				Step 0 (t=1) 			//
	//										//
	//--------------------------------------//

	// Initial Gradient sigma2_t. From eq (186)
	
	dlib::matrix<double,4,1> Gradsigma2;
	
	Gradsigma2(0) = 0.0;							//Dsigma2_r
	Gradsigma2(1) = 2.0 * sigma0 * (1.0 - alpha);	//Dsigma2_sigma
	Gradsigma2(2) = - sigma0 * sigma0;				//Dsigma2_alpha
	Gradsigma2(3) = 0.0;							//Dsigma2_beta

	// Initial Gradient X_t. From eq (193)

	dlib::matrix<double,8,1> GradX;
	GradX = 0.0;
	
	GradX(0) = -eta;							//DX_r
	GradX(5) = 1.0;								//DX_Xbar
	
	// Starting sigma2, X, lambda	
	double sigma2 = sigma0 * sigma0 * (1.0 - alpha);
	double sigma = sqrt(sigma2);

	double X = Xbar;
	double lambda = L(X);						// F(X)
	double FpX = lambda * lambda * exp(-X);		// F'(X)
	double K = K0;
	double mu_t = fma(K, lambda, mu);

	// Initial Jacobian Matrix for psi_t

	dlib::matrix<double,4,8> JacPsi;
	
	JacPsi = 0.0;
	
	JacPsi(0,0) = 1.0;			// 1st row of JacPsi. Eq (180)
	
	JacPsi(1,0) = Gradsigma2(0); //------------------
	JacPsi(1,1) = Gradsigma2(1); // 2nd row, that is 
	JacPsi(1,2) = Gradsigma2(2); // grad(sigma2)
	JacPsi(1,3) = Gradsigma2(3); //------------------

	JacPsi(2,4) = 1.0;			// 3rd row. Eq (180)		

	JacPsi(3,0) = FpX * GradX(0);	// 4th row. Only r_bar and 
	JacPsi(3,5) = FpX * GradX(5);	// Xbar != 0
	
	// Calculation of EI, EJI, EJ2I

	dlib::matrix<double,3,1> Vec_G;
	
	if (K==0.0)
	{	
		Vec_G(0) = lambda;
		Vec_G(1) = lambda;
		Vec_G(2) = 2.0 * lambda;
	}
	else
	{
		double A = (r(0)-mu_t) / sigma + sigma/K; // eq. (135)
		double abs_A = fabs(A);
	
		Vec_G(0) = Rnorm(abs_A);						// EI
		Vec_G(1) = fma(-abs_A,Vec_G(0),1.0);			// EJI
		Vec_G(2) = fma(-abs_A,Vec_G(1),Vec_G(0));		// EJ2
		
		if ((sign(K)*A) < 0)
		{
			Vec_G(0) = 1.0 - Vec_G(0) * exp(-0.5*A*A - 0.91893853320467274178);
			Vec_G(1) = abs_A + Vec_G(1) * exp(-0.5*A*A - 0.91893853320467274178);
			Vec_G(2) = A*A + 1.0 - Vec_G(2) * exp(-0.5*A*A - 0.91893853320467274178);
		}
				
		double den = Vec_G(0) + (1.0 - lambda)/sigma * phi(sign(K)*A) * fabs(K)/lambda;
		
		Vec_G(0) /= den;
		Vec_G(1) /= den;
		Vec_G(2) /= den;
	}
	
	// Initial matrix R -----> eq (201)

	dlib::matrix<double,4,3> R;
	R = 0.0;
		
	R(3,0) = 1.0 / lambda / (1.0 - lambda);	

	if (K==0.0)
		R(2,1) = -(r(0) - mu ) / sigma2;
	else
	{
		R(0,1) = sign(K) / sigma;							
		R(1,1) = sign(K) * (r(0) - mu_t) / sigma2 / sigma;
		R(2,1) = -(r(0) - mu - 2.0 * K * lambda) / sigma / fabs(K);
		R(3,1) = fabs(K) / sigma;
		
		R(1,2) = 0.5 / sigma2;
		R(2,2) = -1.0 / K;
	}	

	// Initial gradient for f ----> eq. (200)

	dlib::matrix<double,4,1> Gradf;
	
	Gradf(0) = (r(0) - mu_t)/sigma2;

	Gradf(1) = Gradf(0);
	Gradf(1) *= Gradf(0);
	Gradf(1) -= 1.0/sigma2;
	Gradf(1) /= 2.0;

	Gradf(2) = Gradf(0);
	Gradf(2) *= lambda;

	Gradf(3) = Gradf(0);
	Gradf(3) *= K;
	Gradf(3) -= 1.0 / (1.0 - lambda);
	
	Gradf += R * Vec_G;

	// Calculation of Gradient of the Loglikelihood. Eq. (177)
	dlib::matrix<double,8,1> GradL_aux = trans(JacPsi) * Gradf; 

	
	//--------------------------------------//
	//										//
	// 		Step 1 -> T-1 (t=2 -> T) 		//
	//										//
	//--------------------------------------//

	for (int t=1; t<T; t++)
	{	
		// Update Jacobian Gradient sigma2_t eq. (181-184)
		Gradsigma2 *= beta;
	
		Gradsigma2(0) -= 2.0 * alpha * (r(t-1)-mu);							//Dsigma2_r
		Gradsigma2(1) += 2.0 * sigma0 * (1.0-alpha-beta);					//Dsigma2_sigma
		Gradsigma2(2) += (r(t-1)-mu) * (r(t-1)-mu) - sigma0 * sigma0;		//Dsigma2_alpha
		Gradsigma2(3) += fma(-sigma0,sigma0,sigma2);						//Dsigma2_beta
		
		// Update sigma2_t
		sigma2 *= beta;
		sigma2 += omega + alpha * (r(t-1)-mu) * (r(t-1)-mu);
		sigma = sqrt(sigma2);
				
		// Update Jacobian Gradient X_t eq. (188-191)
		GradX *= a;
		
		GradX(0) -= eta;			//DX_mu
		GradX(6) += (r(t-1)-mu);	//DX_eta
		GradX(7) += X - Xbar;		//DX_a
		
		// Update X_t 
		X *= a;
		X += (1.0-a)*Xbar+eta*(r(t-1)-mu);
		
		// Update lambda, Fpx...
		lambda = L(X);
		FpX = lambda*lambda*exp(-X);
		K = K0;	
		mu_t = fma(K,lambda,mu);
		
		// Calculation of EI, EJI, EJ2I
		
		if (K==0.0)
		{
			Vec_G(0) = lambda;
			Vec_G(1) = lambda;
			Vec_G(2) = 2.0 * lambda;
		}
		
		else
		{
				
			double A = (r(t) - mu_t) / sigma + sigma/K;
			double abs_A = fabs(A);
	
			Vec_G(0) = Rnorm(abs_A);
			Vec_G(1) = fma(-abs_A,Vec_G(0),1.0);
			Vec_G(2) = fma(-abs_A,Vec_G(1),Vec_G(0));		
		
			if ((sign(K)*A) < 0)
			{
				Vec_G(0) = 1.0 - Vec_G(0) * exp(-0.5*A*A - 0.91893853320467274178);
				Vec_G(1) = abs_A + Vec_G(1) * exp(-0.5*A*A - 0.91893853320467274178);
				Vec_G(2) = A*A + 1.0 - Vec_G(2) * exp(-0.5*A*A - 0.91893853320467274178);
			}
				
			double den = Vec_G(0) + (1.0 - lambda)/sigma * phi(sign(K)*A) * fabs(K)/lambda;
		
			Vec_G(0) /= den;
			Vec_G(1) /= den;
			Vec_G(2) /= den;
		}		
	
		// Update Jacobian Matrix for psi_t 

		// no need to update other elements != 0. eq. (180)
		
		JacPsi(1,0) = Gradsigma2(0); //
		JacPsi(1,1) = Gradsigma2(1); // 2nd row in JacPsi is Sigma2
		JacPsi(1,2) = Gradsigma2(2); //
		JacPsi(1,3) = Gradsigma2(3); //
				
		JacPsi(3,0) = FpX*GradX(0);	//
		JacPsi(3,5) = FpX*GradX(5);	// 4th row is lamba, that is F'(X) * GradX
		JacPsi(3,6) = FpX*GradX(6);	//
		JacPsi(3,7) = FpX*GradX(7);	//

		// Update matrix R eq. (201)
		R = 0.0;
	
		R(3,0) = 1.0/lambda/(1.0-lambda);
	
		if (K==0.0)
			R(2,1) = -(r(t) - mu ) / sigma2;
		else
		{
			R(0,1) = sign(K)/sigma;
			R(1,1) = sign(K)*(r(t)-mu_t)/sigma2/sigma;
			R(2,1) = -(r(t)-mu-2.0*K*lambda)/sigma/fabs(K);
			R(3,1) = fabs(K)/sigma;
	
			R(1,2) = 0.5/sigma2;
			R(2,2) = -1.0/K;
		}
	
		// Update gradient for f eq. (200)
		
		Gradf(0) = (r(t)-mu_t)/sigma2;

		Gradf(1) = Gradf(0);
		Gradf(1) *= Gradf(0);
		Gradf(1) -= 1.0/sigma2;
		Gradf(1) /= 2.0;

		Gradf(2) = Gradf(0);
		Gradf(2) *= lambda;

		Gradf(3) = Gradf(0);
		Gradf(3) *= K;
		Gradf(3) -= 1.0 / (1.0 - lambda);
		
		Gradf += R * Vec_G;
		
		// Calculation of Gradient of the Loglikelihood. Eq. (177)
		GradL_aux += trans(JacPsi) * Gradf; 
	}

	return -GradL_aux/T;
}

dlib::matrix<double> Loglikelihood_hessian(const dlib::matrix<double,0,1>& r, const dlib::matrix<double,0,1>& param){

	const double mu = param(0);
	const double sigma0 = param(1);
	const double alpha = param(2);
	const double beta = param(3);
	const double K0 = param(4);
	const double Xbar = param(5);
	const double eta = param(6);
	const double a = param(7);

	//----------------------------------------
	// Check constraints
	//----------------------------------------
	if (sigma0<0.0)
		std::cout << "Parameter SIGMA0 must be positive" << std::endl; 
		
	if (alpha>1.0 || alpha<0.0)
		std::cout << "Parameter ALPHA must belong to (0,1)" << std::endl; 
	
	if (beta>1.0 || beta<0.0)
		std::cout << "Parameter BETA must belong to (0,1)" << std::endl; 
	
	if (alpha+beta>1.0)
		std::cout << "ALPHA + BETA must be less than one" << std::endl; 
	
	if (K0<0.0)
		std::cout << "Parameter kappa must be positive" << std::endl; 

	if (eta<0.0)
		std::cout << "Parameter ETA must be positive" << std::endl; 
		
	if (a>1.0 || a<0.0)
		std::cout << "Parameter a must belong to (0,1)" << std::endl;

	double omega = sigma0 * sigma0 * (1.0L - alpha - beta);
	
	int T = r.size();
		
	dlib::matrix<double,8,1> GradL_aux;
	
	//int i,j,k,l;

	//--------------------------------------//
	//									    //
	//			     Step 0 			    //
	//									    //
	//--------------------------------------//

	// Initial Gradient sigma2_t. From eq (186)

	dlib::matrix<double,4,1> Gradsigma2;
	
	Gradsigma2(0) = 0.0;							//Dsigma2_r
	Gradsigma2(1) = 2.0* sigma0 * (1.0 - alpha);	//Dsigma2_sigma
	Gradsigma2(2) = -sigma0 * sigma0;				//Dsigma2_alpha
	Gradsigma2(3) = 0.0;							//Dsigma2_beta

	// Initial Gradient X_t. From eq (193)
	
	dlib::matrix<double,8,1> GradX;
	GradX = 0.0;
	
	GradX(0) = -eta;								//DX_r
	GradX(5) = 1.0;								//DX_Xbar
	
	// Starting sigma2, X, lambda	
	double sigma2 = sigma0 * sigma0 * (1.0 - alpha);
	double sigma = sqrt(sigma2);
	
	double X = Xbar;
	double lambda = L(X);						// F(X)
	double FpX = lambda * lambda * exp(-X);	// F'(X)
	double Fp2X = lambda * FpX * expm1(-X);	// F"(X)
	double K = K0;
	double mu_t = fma(K, lambda, mu);

	
	// Initial Jacobian Matrix for psi_t
		
	dlib::matrix<double,4,8> JacPsi;
	
	JacPsi = 0.0;
	
	JacPsi(0,0) = 1.0;			// 1st row of JacPsi. Eq (180)

	JacPsi(1,0) = Gradsigma2(0); 	//------------------
	JacPsi(1,1) = Gradsigma2(1); 	// 2nd row, that is 
	JacPsi(1,2) = Gradsigma2(2); 	// grad(sigma2)
	JacPsi(1,3) = Gradsigma2(3); 	//------------------

	JacPsi(2,4) = 1.0;			// 3rd row. Eq (180)

	JacPsi(3,0) = FpX * GradX(0);	// 4th row. Only r_bar and 
	JacPsi(3,5) = FpX * GradX(5);	// Xbar != 0

	// Initial Hessian Matrix for sigma_t^2
	
	dlib::matrix<double,8,8> Hsigma2;
	Hsigma2 = 0.0;
	
	Hsigma2(0,0) = 2.0 * alpha;					// 
	Hsigma2(1,1) = 2.0 * (1.0 - alpha);			// eq. (218)
	Hsigma2(1,2) = Hsigma2(2,1) = -2.0 * sigma0;	//

	// Initial Hessian Matrix for X_t
	
	dlib::matrix<double,8,8> HX;
	HX = 0.0;
	
	HX(0,6) = HX(6,0) = -1.0L;	// eq. (230) only d(r_bar)d(eta) =! 0
	
	// Initial Hessian Matrix for lambda_t

	dlib::matrix<double,8,8> Hlambda;
	Hlambda = FpX * HX + Fp2X * GradX * trans(GradX); // from eq. (178-179)
		
	// Calculation of EI, EJI, EJ2I, EJ3I and EJ4I
	
	double R0,R1,R2,R3,R4;
	
	if (K==0.0)
	{	
		R0 = lambda;
		R1 = lambda;
		R2 = 2.0 * lambda;
		R3 = 6.0 * lambda;
		R4 = 24.0 * lambda;	
	}
	else
	{
		double A = (r(0) - mu_t) / sigma + sigma/K; // eq. (135)
		double abs_A = fabs(A);
		
		R0 = Rnorm(abs_A); // it shouldn't be Rnorm(sign(K)*A) ??????????? is abs_A = sign(K)*A ?????
		R1 = fma(-abs_A,R0,1.0L);
		R2 = fma(-abs_A,R1,R0);
		R3 = fma(-abs_A,R2,2.0L*R1);
		R4 = fma(-abs_A,R3,3.0L*R2);
		
		if ((sign(K)*A)<0)
		{
			R0 = fma(-R0,exp(-.5L*A*A-.91893853320467274178L),1.0L);
			R1 = fma(R1,exp(-.5L*A*A-.91893853320467274178L),abs_A);
			R2 = fma(-R2,exp(-.5L*A*A-.91893853320467274178L),fma(A,A,1.0L));
			R3 = fma(R3,exp(-.5L*A*A-.91893853320467274178L),abs_A*fma(A,A,3.0L));
			R4 = fma(-R4,exp(-.5L*A*A-.91893853320467274178L),fma(A*A,fma(A,A,6.0L),3.0L));
		}
						
		double den = fma((1.0L - lambda) / sigma, phi(sign(K)*A) * fabs(K) / lambda, R0);
		
		R0 /= den;
		R1 /= den;
		R2 /= den;
		R3 /= den;
		R4 /= den;
	}	
	
	dlib::matrix<double,3,1> Vec_G;	
	
	Vec_G(0) = R0;
	Vec_G(1) = R1;
	Vec_G(2) = R2;
			
	// Initial matrix R -----> eq (201)

	dlib::matrix<double,4,3> R;
	R = 0.0;
		
	R(3,0) = 1.0 / lambda / (1.0 - lambda);
	
	if (K==0.0)
		R(2,1) = -(r(0) - mu ) / sigma2;
	else
	{
		R(0,1) = sign(K) / sigma;							
		R(1,1) = sign(K) * (r(0)-mu_t) / sigma2 / sigma; 		
		R(2,1) = -(r(0) - mu - 2.0 * K * lambda) / sigma / fabs(K);
		R(3,1) = fabs(K) / sigma;

		R(1,2) = 0.5 / sigma2;							
		R(2,2) = -1.0 / K;
	}
	
	// Initial gradient for f ----> eq. (200)
	
	dlib::matrix<double,4,1> Gradf;
	
	Gradf(0) = (r(0) - mu_t) / sigma2;
	
	Gradf(1) = Gradf(0);
	Gradf(1) *= Gradf(0);
	Gradf(1) -= 1.0/sigma2;
	Gradf(1) /= 2.0;

	Gradf(2) = Gradf(0);
	Gradf(2) *= lambda;

	Gradf(3) = Gradf(0);
	Gradf(3) *= K;
	Gradf(3) -= 1.0 / (1.0 - lambda);
	
	Gradf += R * Vec_G;	
		
	// Initial Matrix G
	
	dlib::matrix<double,3,3> G;
	G = 0.0;
	
	G(0,0) = fma(-R0,R0,R0); 			// R0 = E(I)
	G(0,1) = G(1,0)=fma(-R0,R1,R1);		// R1 = E(JI)
	G(0,2) = G(2,0)=fma(-R0,R2,R2);		// R2 = E(J2I)

	G(1,1) = fma(-R1,R1,R2);			
	G(1,2) = G(2,1) = fma(-R1,R2,R3);	// R3 = E(J3I)
	
	G(2,2) = fma(-R2,R2,R4);			// R4 = E(J4I)
	
	// Initial total Hessian Matrix for f ------> eq (231-244)
	
	dlib::matrix<double,4,4> Hf;
	
	Hf(0,0) = -1.0 / sigma2;
	Hf(0,1) = Hf(1,0) = -((r(0)-mu_t)/sigma2+sign(K)/sigma*R1)/sigma2;
	Hf(0,2) = Hf(2,0) = R1/sigma/fabs(K)-lambda/sigma2;
	Hf(0,3) = Hf(3,0) = -K/sigma2;
	Hf(1,1) = (0.5L-R2)/sigma2/sigma2;
	Hf(1,1) -= ((r(0)-mu_t)*((r(0)-mu_t)/sigma2+2.0L*sign(K)*R1/sigma))/pow(sigma2,2);	
	Hf(1,2) = Hf(2,1) = (r(0)-mu_t)/sigma2*(R1/sigma/fabs(K)-lambda/sigma2)+(R2/K - sign(K)*lambda/sigma*R1)/sigma2;
	Hf(1,3) = Hf(3,1) = -K/sigma2*((r(0)-mu_t)/sigma2 + sign(K)*R1/sigma);
	Hf(2,2) = fma(lambda/sigma,fma(2.0/fabs(K),R1,-lambda/sigma),-R2/K/K);
	Hf(2,3) = Hf(3,2) = fma(2.0*sign(K)/sigma,R1,(r(0)-mu-2.0L*K*lambda)/sigma2);
	Hf(3,3) = -K*K/sigma2;
	Hf(3,3) -= R0/(lambda*lambda);
	Hf(3,3) -= (1.0L-R0)/(1.0L-lambda)/(1.0L-lambda);
		
	Hf = Hf + R * G * trans(R); // eq. (298)
	
	// Calculation of the Hessian of L. eq (179)
	
	dlib::matrix<double,8,8> HL_aux;
	HL_aux = trans(JacPsi) * Hf * JacPsi + Gradf(1) * Hsigma2 + Gradf(3) * Hlambda;

	//--------------------------------------//
	//									    //
	//		     Step 1 to T-1 			    //
	//									    //
	//--------------------------------------//

	for (int t=1;t<T;t++)
	{
		// Update Hessian Matrix for sigma_t^2 eq. (207-216)
	
		double beta_t = expm1((t+1)*log1p(beta-1.0L))/(beta-1.0L);
	
		Hsigma2(0,0) = 2.0L * alpha * beta_t; // Hsgima2(0,0) = 2.0L*alpha + beta*Hsigma2(0,0) ??? or UpdateAR1(2.0L*alpha,Hsigma2(0,0),beta,&HS00c) and add HS00c
		Hsigma2(0,2) = Hsigma2(2,0) = -2.0L*(r(t-1)-mu) + Hsigma2(0,2) * beta;
		Hsigma2(0,3) = Hsigma2(3,0) = Gradsigma2(0) + Hsigma2(0,3) * beta;
		Hsigma2(1,1) = 2.0L - Hsigma2(0,0); // UpdateAR1(2.0L*(1.0L-alpha-beta),Hsigma2(1,1),beta,&HS11c)
		Hsigma2(1,2) = Hsigma2(2,1) = -2.0L*sigma0*beta_t; // UpdateAR1(2.0L*sigma0,Hsigma2(1,2),beta,&HS12c)
		Hsigma2(1,3) = Hsigma2(3,1) = 2.0L*alpha*sigma0/pow(1.0L-beta,2)*expm1(fma(t,log1p(beta-1.0L),log1p(t*(1.0L-beta)))); 
		Hsigma2(2,3) = Hsigma2(3,2) = Gradsigma2(2) + Hsigma2(2,3) * beta;
		Hsigma2(3,3) = 2.0L*Gradsigma2(3) + Hsigma2(3,3) * beta;
		
		// Update Hessian Matrix for X_t eq. (219-227)
	
		double a_t = expm1((t+1)*log1p(a-1.0L))/(a-1.0L);
	
		HX(0,6) = HX(6,0)=-a_t; // UpdateAR1(-1.0L,HX(0,6),a,&HX06c)
		HX(0,7) = HX(7,0)=eta/pow(1.0L-a,2)*expm1(fma(t,log1p(a-1.0L),log1p(t*(1.0L-a)))); // UpdateAR1(GradX(0),HX(0,7),a,&HX07c);
		
		// HX(5,7)=HX(7,5)=ExactSum(UpdateAR1(GradX(5),HX(5,7),a,&HX57c),1.0L, &???)
		HX(6,7) = HX(7,6) = GradX(6) + HX(6,7) * a;		
	
		HX(7,7) = 2.0L*GradX(7) + HX(7,7) * a;
		
		// Update Jacobian Gradient sigma2_t eq. (181-184) TODO
		
		sigma2 = omega+alpha*(r(t-1)-mu)*(r(t-1)-mu) + sigma2 * beta; // new sigma2_t
		sigma = sqrt(sigma2);
		
		Gradsigma2(0) = alpha*Hsigma2(0,2);					//Dsigma2_r
		Gradsigma2(1) = 2.0L*sigma0*(1.0L-alpha*beta_t);		//Dsigma2_sigma
		Gradsigma2(2) = fmal(-sigma0,sigma0,sigma2)/alpha;	//Dsigma2_alpha
		Gradsigma2(3) = alpha*Hsigma2(2,3);					//Dsigma2_beta
		
		// Update Jacobian Gradient X_t eq. (188-191) TODO
		
		X = (1.0L-a)*Xbar+eta*(r(t-1)-mu) + X * a;
		lambda = L(X);
		FpX = lambda*lambda*expl(-X);
		Fp2X = lambda*FpX*expm1l(-X);
		
		GradX(0) = -eta*a_t;			//DX_mu
		//GradX(5)=1.0L;			//DX_Xbar
		GradX(6) = (X-Xbar)/eta;		//DX_eta
		GradX(7) = eta*HX(6,7);		//DX_a
		
		// Update Observation point. Calculation of EI, EJI, EJ2I, EJ3I and EJ4I

		K = K0;	
		mu_t = fma(K,lambda,mu);

		if (K==0.0L)
		{	
			R0 = lambda;
			R1 = lambda;
			R2 = 2.0L*lambda;
			R3 = 6.0L*lambda;
			R4 = 24.0L*lambda;		
		}
		else
		{
			double A = (r(t) - mu_t) / sigma + sigma/K;
			double abs_A = fabs(A);
		
			R0 = Rnorm(abs_A);
			R1 = fma(-abs_A,R0,1.0L);
			R2 = fma(-abs_A,R1,R0);
			R3 = fma(-abs_A,R2,2.0L*R1);
			R4 = fma(-abs_A,R3,3.0L*R2);
		
			if ((sign(K)*A)<0)
			{
				R0 = fma(-R0,exp(-.5L*A*A-.91893853320467274178L),1.0L);
				R1 = fma(R1,exp(-.5L*A*A-.91893853320467274178L),abs_A);
				R2 = fma(-R2,exp(-.5L*A*A-.91893853320467274178L),fma(A,A,1.0L));
				R3 = fma(R3,exp(-.5L*A*A-.91893853320467274178L),abs_A*fma(A,A,3.0L));
				R4 = fma(-R4,exp(-.5L*A*A-.91893853320467274178L),fma(A*A,fma(A,A,6.0L),3.0L));
			}
				
			double den = fma((1.0L-lambda)/sigma,phi(sign(K)*A)*fabs(K)/lambda,R0);
		
			R0 /= den;
			R1 /= den;
			R2 /= den;
			R3 /= den;
			R4 /= den;
		}		
	
		// Update Jacobian Matrix for psi_t 

		// no need to update other elements != 0. eq. (180)
		
		JacPsi(1,0) = Gradsigma2(0); //
		JacPsi(1,1) = Gradsigma2(1); // 2nd row in JacPsi is Sigma2
		JacPsi(1,2) = Gradsigma2(2); //
		JacPsi(1,3) = Gradsigma2(3); //
				
		JacPsi(3,0) = FpX*GradX(0);	//
		JacPsi(3,5) = FpX*GradX(5);	// 4th row is lamba, that is F'(X) * GradX
		JacPsi(3,6) = FpX*GradX(6);	//
		JacPsi(3,7) = FpX*GradX(7);	//		
	
		// Update Hessian Matrix for lambda_t eq. (179)
	
		Hlambda = FpX * HX + Fp2X * GradX * trans(GradX); // from eq. (178-179)
	
		// Update matrix R eq. (201)

		R = 0.0L;
	
		R(3,0) = 1.0L/lambda/(1.0L-lambda);
	
		R(0,1) = sign(K)/sigma;
		R(1,1) = sign(K)*(r(t)-mu_t)/sigma2/sigma;
		R(2,1) = -(r(t)-mu-2.0L*K*lambda)/sigma/fabs(K);
		R(3,1) = fabs(K)/sigma;
	
		R(1,2) = 0.5L/sigma2;
		R(2,2) = -1.0/fabs(K);
	
		Vec_G(0) = R0;
		Vec_G(1) = R1;
		Vec_G(2) = R2;
	
		// Update gradient for f eq. (200)
			
		Gradf(0) = (r(t)-mu_t)/sigma2;
		
		Gradf(1) = 0.5L*(r(t)-mu_t)*(r(t)-mu_t)/(sigma2*sigma2);
		Gradf(1) -= 0.5L/sigma2;
	
		Gradf(2) = lambda*(r(t)-mu_t)/sigma2;
		
		Gradf(3) = K*(r(t)-mu_t)/sigma2;
		Gradf(3) -= 1.0L/(1.0L-lambda);
		
		Gradf = Gradf + R * Vec_G;
		
		// Update Matrix G
	
		G(0,0) = fma(-R0,R0,R0);
		G(0,1) = G(1,0)=fma(-R0,R1,R1);
		G(0,2) = G(2,0)=fma(-R0,R2,R2);

		G(1,1) = fma(-R1,R1,R2);
		G(1,2) = G(2,1)=fma(-R1,R2,R3);

		G(2,2) = fma(-R2,R2,R4);
				
		// Update Hessian Matrix for f eq. (231-244)
	
		Hf(0,0) = -1.0L/sigma2;
		Hf(0,1) = Hf(1,0)=-((r(t)-mu_t)/sigma2+sign(K)/sigma*R1)/sigma2;
		Hf(0,2) = Hf(2,0)=R1/sigma/fabs(K)-lambda/sigma2;
		Hf(0,3) = Hf(3,0)=-K/sigma2;
		Hf(1,1) = (0.5L-R2)/sigma2/sigma2;
		Hf(1,1) -= ((r(t)-mu_t)*((r(t)-mu_t)/sigma2+2.0L*sign(K)*R1/sigma))/pow(sigma2,2);	
		Hf(1,2) = Hf(2,1)=(r(t)-mu_t)/sigma2*(R1/sigma/fabs(K)-lambda/sigma2)+(R2/K - sign(K)*lambda/sigma*R1)/sigma2;
		Hf(1,3) = Hf(3,1)=-K/sigma2*((r(t)-mu_t)/sigma2 + sign(K)*R1/sigma);
		Hf(2,2) = fma(lambda/sigma,fma(2.0/fabs(K),R1,-lambda/sigma),-R2/K/K);
		Hf(2,3) = Hf(3,2)=fma(2.0*sign(K)/sigma,R1,(r(t)-mu-2.0L*K*lambda)/sigma2);
		Hf(3,3) = -K*K/sigma2;
		Hf(3,3) -= R0/(lambda*lambda);
		Hf(3,3) -= (1.0L-R0)/(1.0L-lambda)/(1.0L-lambda);
	
		Hf = Hf + R * G * trans(R); // eq. (298)
		
		// Calculation of the Hessian of L. eq (179)
			
		HL_aux = HL_aux + trans(JacPsi) * Hf * JacPsi + Gradf(1) * Hsigma2 + Gradf(3) * Hlambda;
	}

	return -HL_aux/T;

	//--------------------------------------//
	// 	Check Hessian positive-definiteness	//
	//	if not, modify it using some norm 	//
	//--------------------------------------//

	// dlib::matrix<double,8,8> Hess, V, D;
	// Hess = -HL_aux/T;

	// std::cout << "Original Hessian\n" << Hess << std::endl; 

	// dlib::eigenvalue_decomposition<dlib::matrix<double> > eig(Hess);

	// // Eigenvectors
	// V = eig.get_pseudo_v();
	// // Eigenvalues
	// D = eig.get_pseudo_d();
	// // set the threshold to consider an eigenvalue too close to zero
	// double delta = 0.00001;
	// // if (D(0,0)<0.0 || D(1,1)<0.0 || D(2,2)<0.0 || D(3,3)<0.0 || D(4,4)<0.0 || D(5,5)<0.0 || D(6,6)<0.0 || D(7,7)<0.0)
	// if (dlib::min(D) < 0.0)
	// {
	// 	// std::cout << "Hessian is not semi-definite positive" << std::endl;
	// 	// Euclidian norm
	// 	// D += (delta - dlib::min(D))*identity_matrix<double>(8);
	// 	for (int i=0; i<8; i++)
	// 	{
	// 		if (D(i,i)<0)
	// 		{
	// 			// frobenius norm
	// 			D(i,i) = 1e-7;
	// 		}
	// 	}
	// 	// recompute the Hessian that should be positive definite now
	// 	Hess = V*D*trans(V);
	// 	// std::cout << "New Hessian\n" << Hess << std::endl; 

	// }

	// return Hess;
}

//----------------------------------------------------------//
//															//
// Function computing log-likelihood for GARCH(1,1) model	//
//															//
//----------------------------------------------------------//

double garchLogLikelihood(const dlib::matrix<double,0,1>& ret, const dlib::matrix<double,0,1>& B)
// B(0) = rbar 		--> mean-return
// B(1) = sigmabar 	-->	long-time variance
// B(2) = alpha
// B(3) = beta

{
    long n = ret.size();
    column_vector ret1(n);              // De-meaned return
    column_vector ret2(n);              // Return squared
    
    if  ( (B(1)<0.0) || (B(2)<0.0) || (B(3)<0.0) ||(B(2)+B(3)>=1) )
    {
        // std::cout << "Constraints not satisfied = " << trans(B) << std::endl;
        // Penalty for non-permissible parameter values
        return 1e100;
    }

    else
    {
        //---------------------------------//
        //  Construct the log likelihood   //
        //---------------------------------//
        // de-mean the return vector
        ret1 = ret - B(0);
        // squared returns
        ret2 = dlib::pointwise_multiply(ret1, ret1);
		
        long double sigma2 = B(1);
        sigma2 *= B(1);
		sigma2 *=1.0L - B(2);
		
		long double Loglik = 0.0L;
		Loglik += log(sigma2) + ret2(0)/sigma2;
		
        for (int i=1; i<n; ++i)
        {
        	sigma2 *= B(3);
			sigma2 += B(1) * B(1) * (1.0L-B(2)-B(3)) + B(2)*ret2(i-1);
			Loglik += log(sigma2) + ret2(i)/sigma2; // excluding the constants and *(-1)
        }
        
        return (double) (Loglik/(2.0L * n) + 0.91893853320467274178L);
    }
    
}

//----------------------------------------------------------//
//															//
// Function computing gradient of GARCH(1,1) log-likelihood //
//															//
//----------------------------------------------------------//

const dlib::matrix<double,0,1> garchll_derivative(const dlib::matrix<double,0,1>& ret, const dlib::matrix<double,0,1>& B)
// B(0) = rbar 		--> mean-return
// B(1) = sigmabar 	-->	long-time volatility
// B(2) = alpha
// B(3) = beta
{
    
    long n = ret.size();
    column_vector ret1(n);					// De-meaned return
    column_vector ret2(n);                  // Return squared
     
    // check constraints
    if  ( (B(1)<0.0) || (B(2)<0.0) || (B(3)<0.0) ||(B(2)+B(3)>=0.999) )
    {
        // std::cout << "Constraints not satisfied = " << trans(B) << std::endl;
        // if close to the constraints use approximate derivative to increase stability
        return dlib::derivative(garchLogLikelihood, 1e-7)(ret, B);
    }

    else
    {   
        //---------------------------------//
        //  Construct the log likelihood   //
        //---------------------------------//
        // de-mean the return vector
        ret1 = ret - B(0);
        // squared returns
        ret2 = dlib::pointwise_multiply(ret1, ret1);
    
		dlib::matrix<double,0,1> GradL(4);
		dlib::matrix<double,0,1> GradSigma(4);

		long double sigma2 = B(1);
		sigma2 *= B(1);
		sigma2 *= 1.0L - B(2);
		
		long double cc =  1.0L/sigma2 - ret2(0)/(sigma2*sigma2);
		
		GradSigma(0) = 0.0;
		GradSigma(1) = 2*B(1);
		GradSigma(1) *= 1.0 - B(2);
		GradSigma(2) = - B(1);
		GradSigma(2) *= B(1);
		GradSigma(3) = 0.0;
		
		GradL = cc * GradSigma;
		// derivative wrt rbar is different
		GradL(0) -= 2.0 * ret1(0) / sigma2;
				
        for (int i=1; i<n; ++i) {

			GradSigma *= B(3);
			
			GradSigma(0) -=  2.0 * B(2) * ret1(i-1);
			GradSigma(1) += 2*B(1)*(1.0 - B(2) - B(3));
			GradSigma(2) -= B(1)*B(1) - ret2(i-1);
			GradSigma(3) -= B(1)*B(1) - sigma2;
			
			sigma2 *= B(3);
			sigma2 += B(1)*B(1) * (1.0L-B(2)-B(3)) + B(2)*ret2(i-1);
        
			cc =  1.0L/sigma2 - ret2(i)/(sigma2*sigma2);
			
			GradL += cc * GradSigma;
			// derivative wrt rbar is different
			GradL(0) -= 2.0 * ret1(i) / sigma2;
		}	
		
		return GradL/(2*n);
    }
}   


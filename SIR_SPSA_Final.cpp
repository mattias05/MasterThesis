//  Created by Mattias Franchetto on 01/12/16.
//  Copyright (c) 2016 Mattias Franchetto. All rights reserved.

//----------------------------//
//         Input              //
// Time series of log returns //
//----------------------------//

//----------------------------//
//         Output             //
//     Estimated Parameters   //
//    (value, min, max, sd)   //
//----------------------------//

#include <iostream>         // Include in/out function
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include <cstdio>
#include <ctime>
#include <time.h>
#include <algorithm>        // for low_bound and upper_bound
#include <iterator>         // std::begin, std::end
#include <iomanip>          // for setw
//#include <omp.h>
using namespace std;        // The namespace

#ifndef M_PI
    #define M_PI 3.14159265358979323846264L
#endif

// # particles
#define N 1000

// time steps
#define T1 10000

// max number of perturbation iterations 
#define MAX_PERTURBATION_ITER 1000

// max number of parameters iterations 
#define MAX_PARAM_UPDATE_ITER 100000

// sigma min
#define SIGMA_MIN 0.0001L

// sigma min
#define K0_MIN 0.0001L

// Normal cumulative density function:
long double pnorm(long double q, long double mu=0.0L, long double sd=1.0L) {
  const long double SQRTHL = 7.071067811865475244008e-1L;
  long double x, y, z;

  x = (q-mu)/sd * SQRTHL;
  z = abs(x);

  if( z < SQRTHL )
    y = 0.5L + 0.5L * erf(x);
  else
    {
      y = 0.5L * erfc(z);

      if( x > 0.0L )
	y = 1.0L - y;
    }
  return y;
}

// sgn function
long double sign(long double x) {
  if (x >= 0.0L) {
    return 1.0L;
  }
  else {
    return -1.0L;
  }
}

// Normal density:
long double dnorm(long double x, long double mu, long double sd2) {
  // sd2 denotes the variance
  return exp(-.5L*(x-mu)*(x-mu)/sd2-.91893853320467274178L)/sqrt(sd2);
}

// Exponential modified Gaussian density:
long double demg(long double x, long double mu, long double sd, long double k) {
  return 1.0L/abs(k)*exp((x-mu)/k + sd*sd/(2.0L*k*k)) *
    (1.0L - pnorm((sign(k)*((x-mu)/sd + sd/k))));
}

// Function L
long double L(long double x) {
  return (1.0L/(1.0L+exp(-1.0L*x)));
}

// Function computing std deviation
long double standard_deviation(vector<long double>& data, int n)
{
    long double mean_std_dev=0.0L, sum_std_dev=0.0L;
    for (int i=0; i<n; ++i) {
        mean_std_dev += data[i];
    }

    mean_std_dev = mean_std_dev/n;
    for (int i=0; i<n; ++i) {
    	sum_std_dev += (data[i]-mean_std_dev)*(data[i]-mean_std_dev);
    }

    return sqrt(sum_std_dev/(n-1));           
}

int main() {

  random_device rd;
  uint32_t seed=time(NULL);

  // number of parameters
  int p = 8.0L;
  // number of NaN and Not NaN
  int d = 0.0L;
  int f = 0.0L;

  // Reading data from simulated the Jump Diffusion model:
  //--------------------------------------------------------------------------//
  // Length of time series:
  unsigned int Ts = 0;

  vector<long double> r;
  vector<long double> X_real;
  vector<long double> V_real;
  vector<int> chrash_activity;
  vector<long double> chrash_size;

  // open the .csv file containing the time series simulated from the model
  FILE * myFile;
  myFile = fopen("../data/Matsim10000_21112016_8.csv", "r");

  if (myFile!=NULL) {
    cout << "Successfully opened the file \n";

    long double aux_r, aux_X_real, aux_V_real, aux_chrash_size;
    int aux_chrash_activity;

    while (fscanf(myFile, "%Lf,%Lf,%i,%Lf,%Lf\n",
		  &aux_X_real, &aux_V_real, &aux_chrash_activity,
		  &aux_chrash_size, &aux_r) == 5) {
      X_real.push_back (aux_X_real);
      V_real.push_back (aux_V_real);
      chrash_activity.push_back (aux_chrash_activity);
      chrash_size.push_back (aux_chrash_size);
      r.push_back (aux_r);
      Ts++;
    }

    fclose (myFile);
    cout << "Data read \n";
  }
  else {
    cout << "Unable to open the file \n";
    return 0;
  }

  cout << "Ts = " << Ts << endl;
  cout << "Random Seed = " << rd() << endl;
  cout << "Random Seed = " << seed << endl;

  unsigned int T = Ts*N;

  // Compute std_dev(returns) and mean(returns) 
  long double sd_r = standard_deviation(r, Ts);
  long double mean_r = 0.0L;

  for (int i=0; i<Ts; ++i){
  	mean_r += r[i];
  }

  mean_r = mean_r/Ts;

  //------------------------//
  // starting distributions //
  //------------------------//
  normal_distribution<long double> normal_mu(mean_r, sqrtl(2.0L)*sd_r);
  //uniform_real_distribution<long double> unif_mu(-0.18L/250.0L,0.32L/250.0L);
  // with gamma we are simulating s0^2. We will take the square root of them
  gamma_distribution<long double> gamma_sigma0((Ts-1.0L)/2.0L, 2.0L*sd_r*sd_r/(Ts-1.0L));
  //uniform_real_distribution<long double> unif_sigma0(0.10L/sqrtl(250.0L),
                 //0.5L/sqrtl(250.0L));
  uniform_real_distribution<long double> unif_K0(0.03L,0.05L);
  uniform_real_distribution<long double> unif_Xbar(-6.0L,-4.0L);
  uniform_real_distribution<long double> unif_eta(2.0L,4.0L);
  uniform_real_distribution<long double> unif_a(0.99L,1.0L);

  vector<unsigned int> I(T);
  vector<long double> J(T);
  vector<long double> pi(T);
  // due to SIR resampling step
  // at the end all weights are identical and equal to 1/N

  vector<long double> mu(T);
  vector<long double> sigma0(T);
  vector<long double> alpha(T);
  vector<long double> beta(T);
  vector<long double> K0(T);
  vector<long double> Xbar(T);
  vector<long double> eta(T);
  vector<long double> a(T);

  // To check particle filters convergence
  vector<double> FF(Ts);

  // Distributions needed for the computations / algorithm:
  default_random_engine generator;
  generator.seed( seed ); // Seeded differently each time

  exponential_distribution<long double> exponential(1.0L);
  normal_distribution<long double> normal(0.0L,1.0L);
  uniform_real_distribution<long double> unif(0.0L,1.0L);

  cout << "End initialisation" << endl;

  //--------------------------------------------------------------------------//
  //--------------------------------------------------------------------------//

  // Vector containing the first stage weights at time t:
  vector<long double> W(N);

  // Vector containing the normalizing first stage weights at time t
  vector<long double> nW(N);
  // Vector containing the cumulative sum of the normalized first stage weights
  // at time t+1
  vector<long double> Q((N+1));

  // Vector containing the position of the re-sampled states at time t+1
  vector<unsigned int> B(N);

  // Vectors containg the resampled states at time t
  vector<long double> is(N);
  vector<long double> js(N);

  vector<long double> TT((N+1));

  vector<long double> Xhat(T); 
  vector<long double> Vhat(T); 

  vector<long double> mus(N);
  vector<long double> sigma0s(N);
  vector<long double> alphas(N);
  vector<long double> betas(N);
  vector<long double> K0s(N);
  vector<long double> Xbars(N);
  vector<long double> etas(N);
  vector<long double> as(N);

  vector<long double> Xhats(N);
  vector<long double> Vhats(N);  

  // to save parameters at the end
  vector<long double> mu_m(Ts);
  vector<long double> sigma0_m(Ts);
  vector<long double> alpha_m(Ts);
  vector<long double> beta_m(Ts);
  vector<long double> K0_m(Ts);
  vector<long double> Xbar_m(Ts);
  vector<long double> eta_m(Ts);
  vector<long double> a_m(Ts);

  // to save min parameters at the end
  vector<long double> mu_min(Ts);
  vector<long double> sigma0_min(Ts);
  vector<long double> alpha_min(Ts);
  vector<long double> beta_min(Ts);
  vector<long double> K0_min(Ts);
  vector<long double> Xbar_min(Ts);
  vector<long double> eta_min(Ts);
  vector<long double> a_min(Ts);

  // to save max parameters at the end
  vector<long double> mu_max(Ts);
  vector<long double> sigma0_max(Ts);
  vector<long double> alpha_max(Ts);
  vector<long double> beta_max(Ts);
  vector<long double> K0_max(Ts);
  vector<long double> Xbar_max(Ts);
  vector<long double> eta_max(Ts);
  vector<long double> a_max(Ts);

  // to save std deviations 
  vector<long double> mu_sd(Ts);
  vector<long double> sigma0_sd(Ts);
  vector<long double> alpha_sd(Ts);
  vector<long double> beta_sd(Ts);
  vector<long double> K0_sd(Ts);
  vector<long double> Xbar_sd(Ts);
  vector<long double> eta_sd(Ts);
  vector<long double> a_sd(Ts);

  // initialise parameters for +/- perturbation
  vector<long double> mu_plus(N); 
  vector<long double> mu_minus(N);
  vector<long double> sigma0_plus(T); 
  vector<long double> sigma0_minus(T);
  vector<long double> alpha_plus(T);
  vector<long double> alpha_minus(T);
  vector<long double> beta_plus(T);
  vector<long double> beta_minus(T);
  vector<long double> K0_plus(T);
  vector<long double> K0_minus(T);
  vector<long double> Xbar_plus(N);
  vector<long double> Xbar_minus(N);
  vector<long double> eta_plus(N);
  vector<long double> eta_minus(N);
  vector<long double> a_plus(N);
  vector<long double> a_minus(N);
  vector<long double> Xhat_plus(N);
  vector<long double> Xhat_minus(N);
  vector<long double> Vhat_plus(N);
  vector<long double> Vhat_minus(N);
  vector<int> I_plus(N);
  vector<int> I_minus(N);
  vector<long double> J_plus(N);
  vector<long double> J_minus(N);
    
  vector<long double> fhat_plus(N);
  vector<long double> fhat_minus(N);

  // initialize perturbations
  vector<long double> delta(p);

  // initialise gradient approximations 
  vector<long double> grad1(N);
  vector<long double> grad2(N);
  vector<long double> grad3(N);
  vector<long double> grad4(N);
  vector<long double> grad5(N);
  vector<long double> grad6(N);
  vector<long double> grad7(N);
  vector<long double> grad8(N);

  // Timing the SIR algorithm;
  clock_t start1;
  double duration1;
  duration1 = 0;

  start1 = clock();

  //--------------------------------------------------------------------------//
  //--------------------------------------------------------------------------//

  // Initialisation, sampling initial states :
  bernoulli_distribution bernoulli_init(0.00669); // with X0 = Xbar = -5 should be 0.00669 it was 0.0006
  exponential_distribution<long double> exponential_init(1.0L); // it was 20.0L

  for (unsigned int n=0; n<N; n++) {

    // Initial sampling of the states:
    mu[n] = normal_mu(generator);
    //mu[n] = unif_mu(generator);
    sigma0[n] = gamma_sigma0(generator);
    sigma0[n] = sqrtl(sigma0[n]);
    //sigma0[n] = unif_sigma0(generator);
    long double E1 = exponential(generator);
    long double E2 = exponential(generator);
    long double E3 = exponential(generator);
    long double se = E1 + E2 + E3;
    alpha[n] = E1 / se;
    beta[n] = E2 / se;
    K0[n] = unif_K0(generator);
    Xbar[n] = unif_Xbar(generator);
    eta[n] = unif_eta(generator);
    a[n] = unif_a(generator);

    // check the constraints
    if(sigma0[n] <= 0.0L || alpha[n] + beta[n] >= 1.0L || alpha[n] <= 0.0L || beta[n] <= 0.0L || K0[n] <= 0.0L){
      return 0;
    }

    // Initialization of mispricing and volatility
    // for each parameter value:
    Xhat[n] = Xbar[n];
    Vhat[n] = sigma0[n] * sigma0[n];
    
    bernoulli_distribution bernoulli_start(L(Xhat[n]));
  	I[n] = bernoulli_start(generator);
  	if (I[n] == 1) {
  		J[n] = exponential_init(generator);
  	}
    pi[n] = 1.0L/N;

    // compute the parameters
    mu_m[0] += 1.0L/N * mu[n];
  	sigma0_m[0] += 1.0L/N * sigma0[n];
  	alpha_m[0] += 1.0L/N * alpha[n];
  	beta_m[0] += 1.0L/N * beta[n];
  	K0_m[0] += 1.0L/N * K0[n];
  	Xbar_m[0] += 1.0L/N * Xbar[n];
  	eta_m[0] += 1.0L/N * eta[n];
  	a_m[0] += 1.0L/N * a[n];

  }

  mu_min[0] = *min_element(begin(mu),begin(mu)+N);
  mu_max[0] = *max_element(begin(mu),begin(mu)+N);
  sigma0_min[0] = *min_element(begin(sigma0),begin(sigma0)+N);
  sigma0_max[0] = *max_element(begin(sigma0),begin(sigma0)+N);
  alpha_min[0] = *min_element(begin(alpha),begin(alpha)+N);
  alpha_max[0] = *max_element(begin(alpha),begin(alpha)+N);
  beta_min[0] = *min_element(begin(beta),begin(beta)+N);
  beta_max[0] = *max_element(begin(beta),begin(beta)+N);
  K0_min[0] = *min_element(begin(K0),begin(K0)+N);
  K0_max[0] = *max_element(begin(K0),begin(K0)+N);
  Xbar_min[0] = *min_element(begin(Xbar),begin(Xbar)+N);
  Xbar_max[0] = *max_element(begin(Xbar),begin(Xbar)+N);
  eta_min[0] = *min_element(begin(eta),begin(eta)+N);
  eta_max[0] = *max_element(begin(eta),begin(eta)+N);
  a_min[0] = *min_element(begin(a),begin(a)+N);
  a_max[0] = *max_element(begin(a),begin(a)+N);

  for (unsigned int n=0; n<N; n++) {
    mu_sd[0] += 1.0L/(N-1) * (mu[n]-mu_m[0]) * (mu[n]-mu_m[0]);
    sigma0_sd[0] += 1.0L/(N-1) * (sigma0[n]-sigma0_m[0]) * (sigma0[n]-sigma0_m[0]);
    alpha_sd[0] += 1.0L/(N-1) * (alpha[n]-alpha_m[0]) * (alpha[n]-alpha_m[0]);
    beta_sd[0] += 1.0L/(N-1) * (beta[n]-beta_m[0]) * (beta[n]-beta_m[0]);
    K0_sd[0] += 1.0L/(N-1) * (K0[n]-K0_m[0]) * (K0[n]-K0_m[0]);
    Xbar_sd[0] += 1.0L/(N-1) * (Xbar[n]-Xbar_m[0]) * (Xbar[n]-Xbar_m[0]);
    eta_sd[0] += 1.0L/(N-1) * (eta[n]-eta_m[0]) * (eta[n]-eta_m[0]);
    a_sd[0] += 1.0L/(N-1) * (a[n]-a_m[0]) * (a[n]-a_m[0]);

  }
  
  mu_sd[0] = sqrtl(mu_sd[0]);
  sigma0_sd[0] = sqrtl(sigma0_sd[0]);
  alpha_sd[0] = sqrtl(alpha_sd[0]);
  beta_sd[0] = sqrtl(beta_sd[0]);
  K0_sd[0] = sqrtl(K0_sd[0]);
  Xbar_sd[0] = sqrtl(Xbar_sd[0]);
  eta_sd[0] = sqrtl(eta_sd[0]);
  a_sd[0] = sqrtl(a_sd[0]);

  cout << "Initial sampling done" << endl;

  // The log likelihood sum for the given parameters:
  long double lls = 0.0L;

  // SPSA parameters
  long double SPSA_a; long double SPSA_A; long double SPSA_alpha;
  long double c_SPSA[] = {0.000001L, 0.000001L, 0.000001L, 0.000001L, 0.000001L, 0.001L, 0.000001L, 0.0005L};
  long double c1_SPSA[] = {0,0,0,0,0,0,0,0};
  long double gamma_SPSA[] = {0.000001L, 0.000001L, 0.000001L, 0.000001L, 0.000001L, 0.0005L, 0.0001L, 0.00001L};
  long double SPSA_r; long double SPSA_c; long double SPSA_c0;
  long double SPSA_ct; long double SPSA_gammat; long double SPSA_ct0; long double SPSA_gammat0;

  // SPSA parameters (that have to be calibrated)
  SPSA_c = 0.000001L; SPSA_c0 = 0.00001L; SPSA_A = 100000.0L; SPSA_alpha = 0.602L;
  SPSA_r = 0.101L; SPSA_a = 0.016L;

  cout << "Starting SIR particle filter with SPSA" << endl;

  // Time loop
  for (unsigned int t=0; t<(T-N); t+=N){

    for (unsigned int n=0; n<N; n++) {
      Xhat[n+t+N] = (1.0L - a[n+t]) * Xbar[n+t] + a[n+t] * Xhat[n+t] +
        eta[n+t] * (r[t/N] - mu[n+t]);
      Vhat[n+t+N] = sigma0[n+t] * sigma0[n+t] *							
          (1.0L - alpha[n+t] - beta[n+t]) + alpha[n+t] *
            pow ((r[t/N] - mu[t+n]), 2.0) + beta[n+t] * Vhat[n+t];

  		bernoulli_distribution bernoulli1(L(Xhat[n+t+N]));
  		I[n+t+N] = bernoulli1(generator);
  		if (I[n+t+N] == 1) {
  			J[n+t+N] = exponential(generator);
  		}
    }

    //--------------------------------
    // START SPSA
    //--------------------------------

    // SPSA gain sequences
    SPSA_ct = SPSA_c / pow(((t+N)/N),SPSA_r);
    for(int j=0; j<p; j++){
      c1_SPSA[j] = c_SPSA[j] / pow(((t+N)/N),SPSA_r);
    }
    // Cost function evaluation 
    // Generate a simultaneous perturbation vector
    bernoulli_distribution bernoulli_delta(0.5L);

    int v = 0.0L;

    // Parallelize???
    //#pragma omp parallel for 

    //{

    for (int n=0; n<N; n++){
	    do {
		    int z = 0.0L;
		    do {
			    for (int d=0; d<p; d++) {
  					if (bernoulli_delta(generator)) { 
  						delta[d] = 1.0L;
  					}
  					else {
  						delta[d] = -1.0L;
  					}
			    }

          // ---------------------------------------------------------------//
			    // Compute the perturbed parameter particle (theta+) and (theta-) //
          // ---------------------------------------------------------------//

  				mu_plus[n] = mu[n+t] + c1_SPSA[0] * delta[0];// / 10.0L;
  				mu_minus[n] = mu[n+t] - c1_SPSA[0] * delta[0];// / 10.0L;

  				sigma0_plus[n] = sigma0[n+t] + c1_SPSA[1] * delta[1];// / 100.0L; // /10.0L
  				sigma0_minus[n] = sigma0[n+t] - c1_SPSA[1] * delta[1];// / 100.0L; // /10.0L

  				alpha_plus[n] = alpha[n+t] + c1_SPSA[2] * delta[2];// / 100.0L;
  				beta_plus[n] = beta[n+t] + c1_SPSA[3] * delta[3];// / 100.0L;

  				alpha_minus[n] = alpha[n+t] - c1_SPSA[2] * delta[2];// / 100.0L;
  				beta_minus[n] = beta[n+t] - c1_SPSA[3] * delta[3];// / 100.0L;

  				K0_plus[n] = K0[n+t] + c1_SPSA[4] * delta[4];// / 10.0L;
  				K0_minus[n] = K0[n+t] - c1_SPSA[4] * delta[4];// / 10.0L;

  				Xbar_plus[n] = Xbar[n+t] + c1_SPSA[5] * delta[5];// / 10.0L;
  				Xbar_minus[n] = Xbar[n+t] - c1_SPSA[5] * delta[5];// / 10.0L;

  				eta_plus[n] = eta[n+t] + c1_SPSA[6] * delta[6];// / 10.0L;
  				eta_minus[n] = eta[n+t] - c1_SPSA[6] * delta[6];// / 10.0L;

  				a_plus[n] = a[n+t] + c1_SPSA[7] * delta[7];// / 10.0L;
  				a_minus[n] = a[n+t] - c1_SPSA[7] * delta[7];// / 10.0L;

			    z++;

			    if(z == MAX_PERTURBATION_ITER){
			    	if(sigma0_plus[n] <= 0.0L){
			    		sigma0_plus[n] = sigma0[n+t];
			    		//cout << "Sigma plus " << sigma0[n+t] << endl;
			    	}
			    	if(sigma0_minus[n] <= 0.0L){
			    		sigma0_minus[n] = sigma0[n+t];
			    		//cout << "Sigma minus " << sigma0[n+t] << endl;
			    	}
			    	if(alpha_plus[n] <= 0.0L){
			    		alpha_plus[n] = alpha[n+t];
			    		//cout << "Alpha plus " << alpha[n+t] << endl;
			    	}
			    	if(alpha_minus[n] <= 0.0L){
			    		alpha_minus[n] = alpha[n+t];
			    		//cout << "alpha minus " << alpha[n+t] << endl;
			    	}
			    	if(beta_plus[n] <= 0.0L){
			    		beta_plus[n] = beta[n+t];
			    		//cout << "beta plus " << beta[n+t]<< endl;
			    	}
			    	if(beta_minus[n] <= 0.0L){
			    		beta_minus[n] = beta[n+t];
			    		//cout << "beta minus " << beta[n+t] << endl;
			    	}
			    	if(alpha_minus[n] + beta_minus[n] >= 1.0L){
			    		alpha_minus[n] = alpha[n+t];
			    		beta_minus[n] = beta[n+t];
			    		//cout << "alpha + beta minus " << alpha[n+t] + beta[n+t] << endl;
			    	}
			    	if(alpha_plus[n] + beta_plus[n] >= 1.0L){
			    		alpha_plus[n] = alpha[n];
			    		beta_plus[n] = beta[n];
			    		//cout << "alpha + beta plus " << alpha[n+t] + beta[n+t] << endl;
			    	}
            if(K0_plus[n] <= 0.0L){
              K0_plus[n] = K0[n];
              //cout << "K plus " << K0[n+t] << endl;
            }
            if(K0_minus[n] <= 0.0L){
              K0_minus[n] = K0[n];
              //cout << "K minus " << K0[n+t] << endl;
            }
			    }

			    if(z > MAX_PERTURBATION_ITER){
			    	cout << "Too many perturbation iterations" << endl;
			    	return 0;
          }

        // check constraints
        } while (sigma0_minus[n] <= 0.0L || sigma0_plus[n] <= 0.0L || alpha_plus[n] + beta_plus[n] >= 1.0L || alpha_minus[n] + beta_minus[n] >= 1.0L || 
				alpha_plus[n] <= 0.0L || beta_plus[n] <= 0.0L || alpha_minus[n] <= 0.0L || beta_minus[n] <= 0.0L || K0_minus[n] <= 0.0L || K0_plus[n] <= 0.0L);

		    // Sample L_i+ and L_i-
		    Xhat_plus[n] = (1.0L - a_plus[n]) * Xbar_plus[n] + 
          a_plus[n] * Xhat[n+t] + eta_plus[n] * (r[t/N] - mu_plus[n]);

        Vhat_plus[n] = sigma0_plus[n] * sigma0_plus[n] *
          (1.0L - alpha_plus[n] - beta_plus[n]) + alpha_plus[n] *
            pow ((r[t/N] - mu_plus[n]), 2.0) + beta_plus[n] * Vhat[n+t];

        Xhat_minus[n] = (1.0L - a_minus[n]) * Xbar_minus[n] + 
          a_minus[n] * Xhat[n+t] + eta_minus[n] * (r[t/N] - mu_minus[n]);

        Vhat_minus[n] = sigma0_minus[n] * sigma0_minus[n] *
          (1.0L - alpha_minus[n] - beta_minus[n]) + alpha_minus[n] *
            pow ((r[t/N] - mu_minus[n]), 2.0) + beta_minus[n] * Vhat[n+t]; 

    		bernoulli_distribution bernoulli1(L(Xhat_plus[n]));
    		I_plus[n] = bernoulli1(generator);
    		if (I_plus[n] == 1) {
      		J_plus[n] = exponential(generator);
    		}

    		bernoulli_distribution bernoulli2(L(Xhat_minus[n]));
    		I_minus[n] = bernoulli2(generator);
    		if (I_minus[n] == 1) {
      		J_minus[n] = exponential(generator);
    		}
    
  	    // Evaluate cost function fhat+ and fhat-
  	  	fhat_plus[n] = dnorm(r[(t+N)/N], 
    			(mu_plus[n] + K0_plus[n] * L(Xhat_plus[n]) - K0_plus[n] * J_plus[n] * I_plus[n]), 
    			Vhat_plus[n]);
    		fhat_minus[n] = dnorm(r[(t+N)/N], 
    			(mu_minus[n] + K0_minus[n] * L(Xhat_minus[n]) - K0_minus[n] * J_minus[n] * I_minus[n]), 
    			Vhat_minus[n]);
    	
        // Gradient approximation 
    		grad1[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[0] * delta[0]);
    		grad2[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[1] * delta[1]);
    		grad3[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[2] * delta[2]);
    		grad4[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[3] * delta[3]);
    		grad5[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[4] * delta[4]);
    		grad6[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[5] * delta[5]);
    		grad7[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[6] * delta[6]);
    		grad8[n] = (fhat_plus[n] - fhat_minus[n]) / (2.0L * c1_SPSA[7] * delta[7]);
    
        //---------------------
        // Parameter update 
        //---------------------        
    		mu[n+t+N] = mu[n+t] + gamma_SPSA[0] * grad1[n];
    		sigma0[n+t+N] = sigma0[n+t] + gamma_SPSA[1] * grad2[n];

        if(sigma0[n+t+N] <= 0.0L){
        	sigma0[n+t+N] = SIGMA_MIN;
        }

    		alpha[n+t+N] = alpha[n+t] + gamma_SPSA[2] * grad3[n];
    		beta[n+t+N] = beta[n+t] + gamma_SPSA[3] * grad4[n];
    		K0[n+t+N] = K0[n+t] + gamma_SPSA[4] * grad5[n];

        if(K0[n+t+N] <= 0.0L){
          K0[n+t+N] = K0_MIN;
        }

    		Xbar[n+t+N] = Xbar[n+t] + gamma_SPSA[5] * grad6[n];
    		eta[n+t+N] = eta[n+t] + gamma_SPSA[6] * grad7[n];
    		a[n+t+N] = a[n+t] + gamma_SPSA[7] * grad8[n];

    		v++; // increment iter

    		if(v > MAX_PARAM_UPDATE_ITER){
  	    	cout << "Too many parameter update iterations" << endl;
  	    	return 0;
    	  }

      } while (sigma0[n+t+N] <= 0.0L || alpha[n+t+N] + beta[n+t+N] >= 1.0L || alpha[n+t+N] <= 0.0L || beta[n+t+N] <= 0.0L || K0[n+t+N] <= 0.0L);
    }

    // calculating the mean of the log-returns including new
    // observation at time t:
    for (int n=0; n<N; n++) {
      W[n] = dnorm(r[(t+N)/N],
       (mu[n+t+N] + K0[n+t+N] * L(Xhat[n+t+N]) - 
        K0[n+t+N] * J[n+t+N] * I[n+t+N]) ,
       Vhat[n+t+N]);
    }

    //--------------------------------
    // END SPSA
    //--------------------------------

    // Resampling --> First stage resampling from the discrete distribution
    // {L_{t+1}, W_{t+1}} using algorithm in multinomialsampling.pdf

    // Normalizing the weights:
    long double s1;
    s1 = 0.0L;
    for (int n=0; n<N; n++) {
      s1 += W[n];
    }

    // Resampling step
    for (unsigned int n=0; n<N; n++) {
      nW[n] = W[n]/s1;
      // Calculating the cumulative sum of the normalized weights:
      // corresponding to Q:
      if (n == 0) {
        Q[n] = 0.0L;
        TT[n] = exponential(generator);
      }
      if (n > 0) {
        Q[n] = Q[n-1] + nW[n-1];
        TT[n] = TT[n-1] + exponential(generator);
      }
    }

    Q[N] = Q[(N-1)] + nW[(N-1)];
    TT[N] = TT[(N-1)] + exponential(generator);

    unsigned int i=0; unsigned int j=1;
    while (i < N) {
  		if ( TT[i] < (Q[j] * TT[N])) {

        // Resample indexes for the states (I, J)
  			B[i] = (j-1);
  			is[i] = I[(j-1)+t+N];
  			js[i] = J[(j-1)+t+N];

        // Resample indexes for the parameters
  			mus[i] = mu[(j-1)+t+N];
  			sigma0s[i] = sigma0[(j-1)+t+N];
  			alphas[i] = alpha[(j-1)+t+N];
  			betas[i] = beta[(j-1)+t+N];
  			K0s[i] = K0[(j-1)+t+N];
  			Xbars[i] = Xbar[(j-1)+t+N];
  			etas[i] = eta[(j-1)+t+N];
  			as[i] = a[(j-1)+t+N];

        // Resample indexes for mispricing index and volatility (Xhat, Vhat)
        Xhats[i] = Xhat[(j-1)+t+N];
        Vhats[i] = Vhat[(j-1)+t+N];

  			i++;
  		}
  		else {
        j++;
    	}
    }
    
    // resample (I,J) (parameters) (Xhat,Vhat)
  	for (unsigned int n=0; n<N; n++) {
  		I[n+t+N] = is[n];
  		J[n+t+N] = js[n];

  		if(n==0){
  			d++;
  		}

  		mu[n+t+N] = mus[n];
  		sigma0[n+t+N] = sigma0s[n];
  		alpha[n+t+N] = alphas[n];
  		beta[n+t+N] = betas[n];
  		K0[n+t+N] = K0s[n];
  		Xbar[n+t+N] = Xbars[n];
  		eta[n+t+N] = etas[n];
  		a[n+t+N] = as[n];

      Xhat[n+t+N] = Xhats[n];
      Vhat[n+t+N] = Vhats[n];
  	}
    
    // Computing the log likelihood at each time step
    long double llst = 0.0L;

    for (unsigned int n=0; n<N; n++) {
    // Using the unnormalised probability weights at time t
    	llst += W[n];
    }
    
    long double ss;
    for (unsigned int n=0; n<N; n++) {
  		pi[n+t+N] = W[n];
  		ss = pi[n+t+N]/llst;
  		pi[n+t+N] = ss;
  	}

    // Updating the complete log-likelihood of the full time series:
    if (isnan(llst) == 1) { 
      // No need to update the log-likelihood
      cout << "Some weights equal to NaN" << endl;
      cout << "NaN log likelihood value ! llst = " << llst << "\n";
      return 0;
      
    }
    else { 
      lls += log(llst);
    }

    cout << "log-likelihood = " << lls << endl;

/*
    // Checking convergence of particle filters:
    double SC;
    SC = 0;
    for (unsigned int j=0; j<N; j++) {
      SC += pi[j+t] * pnorm(r[(t+N)/N],
        mu + K0 * L(X) - K0 * J[j] * I[j],
          sqrtl(V));
    }
    
    FF[t/N] = SC;
 */

    //--------------------------------------------------------------------------//
    // SAVING THE DATA 
    //--------------------------------------------------------------------------//

    //Means
    for (unsigned int n=0; n<N; n++) {
			mu_m[(t+N)/N] += 1.0L/N * mu[t+n+N];
			sigma0_m[(t+N)/N] += 1.0L/N * sigma0[t+n+N];
			alpha_m[(t+N)/N] += 1.0L/N * alpha[t+n+N];
			beta_m[(t+N)/N] += 1.0L/N * beta[t+n+N];
			K0_m[(t+N)/N] += 1.0L/N * K0[t+n+N];
			Xbar_m[(t+N)/N] += 1.0L/N * Xbar[t+n+N];
			eta_m[(t+N)/N] += 1.0L/N * eta[t+n+N];
			a_m[(t+N)/N] += 1.0L/N * a[t+n+N];
    }

    //Min/Max
    mu_min[(t+N)/N] = *min_element(begin(mu)+t+N+1,begin(mu)+t+2*N);
    mu_max[(t+N)/N] = *max_element(begin(mu)+t+N+1,begin(mu)+t+2*N);
    sigma0_min[(t+N)/N] = *min_element(begin(sigma0)+t+N+1,begin(sigma0)+t+2*N);
    sigma0_max[(t+N)/N] = *max_element(begin(sigma0)+t+N+1,begin(sigma0)+t+2*N);
    alpha_min[(t+N)/N] = *min_element(begin(alpha)+t+N+1,begin(alpha)+t+2*N);
    alpha_max[(t+N)/N] = *max_element(begin(alpha)+t+N+1,begin(alpha)+t+2*N);
    beta_min[(t+N)/N] = *min_element(begin(beta)+t+N+1,begin(beta)+t+2*N);
    beta_max[(t+N)/N] = *max_element(begin(beta)+t+N+1,begin(beta)+t+2*N);
    K0_min[(t+N)/N] = *min_element(begin(K0)+t+N+1,begin(K0)+t+2*N);
    K0_max[(t+N)/N] = *max_element(begin(K0)+t+N+1,begin(K0)+t+2*N);
    Xbar_min[(t+N)/N] = *min_element(begin(Xbar)+t+N+1,begin(Xbar)+t+2*N);
    Xbar_max[(t+N)/N] = *max_element(begin(Xbar)+t+N+1,begin(Xbar)+t+2*N);
    eta_min[(t+N)/N] = *min_element(begin(eta)+t+N+1,begin(eta)+t+2*N);
    eta_max[(t+N)/N] = *max_element(begin(eta)+t+N+1,begin(eta)+t+2*N);
    a_min[(t+N)/N] = *min_element(begin(a)+t+N+1,begin(a)+t+2*N);
    a_max[(t+N)/N] = *max_element(begin(a)+t+N+1,begin(a)+t+2*N);

    //Std Dev
    for (unsigned int n=0; n<N; n++) {
      mu_sd[(t+N)/N] += 1.0L/(N-1) * (mu[t+n+N]-mu_m[(t+N)/N]) * (mu[t+n+N]-mu_m[(t+N)/N]);
      sigma0_sd[(t+N)/N] += 1.0L/(N-1) * (sigma0[t+n+N]-sigma0_m[(t+N)/N]) * (sigma0[t+n+N]-sigma0_m[(t+N)/N]);
      alpha_sd[(t+N)/N] += 1.0L/(N-1) * (alpha[t+n+N]-alpha_m[(t+N)/N]) * (alpha[t+n+N]-alpha_m[(t+N)/N]);
      beta_sd[(t+N)/N] += 1.0L/(N-1) * (beta[t+n+N]-beta_m[(t+N)/N]) * (beta[t+n+N]-beta_m[(t+N)/N]);
      K0_sd[(t+N)/N] += 1.0L/(N-1) * (K0[t+n+N]-K0_m[(t+N)/N]) * (K0[t+n+N]-K0_m[(t+N)/N]);
      Xbar_sd[(t+N)/N] += 1.0L/(N-1) * (Xbar[t+n+N]-Xbar_m[(t+N)/N]) * (Xbar[t+n+N]-Xbar_m[(t+N)/N]);
      eta_sd[(t+N)/N] += 1.0L/(N-1) * (eta[t+n+N]-eta_m[(t+N)/N]) * (eta[t+n+N]-eta_m[(t+N)/N]);
      a_sd[(t+N)/N] += 1.0L/(N-1) * (a[t+n+N]-a_m[(t+N)/N]) * (a[t+n+N]-a_m[(t+N)/N]);

    }
    
    mu_sd[(t+N)/N] = sqrtl(mu_sd[(t+N)/N]);
    sigma0_sd[(t+N)/N] = sqrtl(sigma0_sd[(t+N)/N]);
    alpha_sd[(t+N)/N] = sqrtl(alpha_sd[(t+N)/N]);
    beta_sd[(t+N)/N] = sqrtl(beta_sd[(t+N)/N]);
    K0_sd[(t+N)/N] = sqrtl(K0_sd[(t+N)/N]);
    Xbar_sd[(t+N)/N] = sqrtl(Xbar_sd[(t+N)/N]);
    eta_sd[(t+N)/N] = sqrtl(eta_sd[(t+N)/N]);
    a_sd[(t+N)/N] = sqrtl(a_sd[(t+N)/N]);

  } // end of time loop

  cout << "log-likelihood = " << lls << endl;
  cout << "# NaN " << f << endl;
  cout << "# not NaN " << d << endl;

  duration1 += (clock() - start1) / (double) CLOCKS_PER_SEC;
  cout << duration1 << " seconds"  << endl;

  //--------------------------------------------------------------------------//
  // SAVING THE DATA 
  //--------------------------------------------------------------------------//

  ofstream myfile_sim;
  myfile_sim.open ("./Saved_csv_new/10000time1000part2711_Adapt3.csv");
  for (unsigned int i=0; i<Ts; i++) {
    myfile_sim << i << ","
      << mu_m[i] << "," << mu_min[i] << "," << mu_max[i] << "," << mu_sd[i] << ","
      << sigma0_m[i] << "," << sigma0_min[i] << "," << sigma0_max[i] << "," << sigma0_sd[i] << ","
      << alpha_m[i] << "," << alpha_min[i] << "," << alpha_max[i] << "," << alpha_sd[i] << ","
      << beta_m[i] << "," << beta_min[i] << "," << beta_max[i] << "," << beta_sd[i] << ","
      << K0_m[i] << "," << K0_min[i] << "," << K0_max[i] << "," << K0_sd[i] << ","
      << Xbar_m[i] << "," << Xbar_min[i] << "," << Xbar_max[i] << "," << Xbar_sd[i] << ","
      << eta_m[i] << "," << eta_min[i] << "," << eta_max[i] << "," << eta_sd[i] << ","
      << a_m[i] << "," << a_min[i] << "," << a_max[i] << "," << a_sd[i]
      << "\n";
  }
  myfile_sim.close();

  /*  
  //--------------------------------------------------------------------------//
  // Saving the convergence data:
  ofstream myfile_sim2;
  myfile_sim2.open ("convergenceSIR.csv");

  for (unsigned int i=0; i<Ts; i++) {
    myfile_sim2 << FF[i] << "\n";
  }

  myfile_sim2.close();
  //--------------------------------------------------------------------------//
  */

  return 0;
}



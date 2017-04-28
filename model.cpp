// Code to simulate the model

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
using namespace std;       // The namespace

double L(double x) {
  return (1.0/(1.0+exp(-1.0*x)));
}

int main() {
  random_device rd;
  uint32_t seed=time(NULL);

  int T;
  // Length of time series:
  T = 10000;

  // Distributions needed for the computations / algorithm:
  default_random_engine generator;
  generator.seed( seed ); // Seeded differently each time

  exponential_distribution<double> exponential(1.0);
  normal_distribution<double> normal(0.0,1.0);

  // Exact values:
  double r_bar; double k; double X_bar; double eta;
  double vhat; double alpha; double beta; double a;
  r_bar = 0.00028; k = 0.04; X_bar = -5.0; eta = 3.0;
  vhat = (0.25*0.25)/250; alpha = 0.05; beta = 0.94; a = 0.996;

  //------------------------------------//
  // Simulate the jump-diffusion model  //
  //------------------------------------//
  vector<double> r_real(T+1);
  vector<double> X_real(T+1);
  vector<double> V_real(T+1);
  vector<int> I_real(T+1);
  vector<double> J_real(T+1);

  // Initialisation:
  X_real[0] = -4.3;
  V_real[0] = 0.001;
  bernoulli_distribution bernoulli(L(X_real[0]));
  I_real[0] = bernoulli(generator);
  J_real[0] = I_real[0] * k * exponential(generator);

  r_real[0] = r_bar + k * L(X_real[0]) + sqrt(V_real[0]) *
    normal(generator) - J_real[0] * I_real[0];
  cout << "Nrm generator " << normal (generator) << "\t" << r_bar << "\t" << vhat << "\t" << alpha << "\t" << beta 
    << "\t" << k << "\t" << X_bar << "\t" << eta << "\t" << a << endl;

  for (int i=0; i<T; i++) {
    X_real[i+1] = (1-a) * X_bar + a * X_real[i] + eta * (r_real[i] - r_bar);
    V_real[i+1] = vhat * (1.0 - alpha - beta) + alpha *
      (r_real[i] - r_bar)*(r_real[i] - r_bar) + beta * V_real[i];
    bernoulli_distribution bernoulli( L(X_real[i+1]) );

    I_real[i+1] = bernoulli(generator);
    J_real[i+1] = I_real[i+1] * k * exponential(generator); // OBS

    r_real[i+1] = r_bar + k * L(X_real[i+1]) + sqrt(V_real[i+1]) * normal(generator) -
      J_real[i+1] * I_real[i+1];
  }

  // Saving the data:
  ofstream myfile_sim;
  myfile_sim.open ("Test.csv");
  for (int i=0; i<=T; i++) {
    myfile_sim << X_real[i] << "," << V_real[i] <<  ","
	       << I_real[i] << "," << J_real[i] << ","
	       << r_real[i] << "\n";
  }
  myfile_sim.close();
    
  return 0;
}

data {
  
  //number of data points
  int<lower=1> N;            //number of observations
  
  //outcomes
  int gj[N];                //juvenile groomed
  int mg[N];                //juvenile groomed by mom
  
  //predictors
  real age[N];
  real elo[N];
  int sex[N];

  // grouping factors
  int<lower=1> TM;               //number of moms
  int<lower=1,upper=TM> troop_mom[N];  //mom id
  
  int<lower=1> TMJ;                //number of juveniles
  int<lower=1,upper=TMJ> troop_mom_juve[N];  //juve id
  
  int<lower=1> R;                //number of troops
  int<lower=1,upper=R> troop[N];  //troop id
}

parameters {
  
  //predictors
  vector[8] beta;                 // fixed-effects parameters
  
  
  //level 1: variation in random intercepts
  
  real r_troop_j[R]; //random intercept for troop groomed or not
  real r_troop_m[R]; //random intercept for troop if mother groomed or not
  
  real r_troop_mom_j[TM]; //random intercept for mom on groomed or not
  real r_troop_mom_m[TM]; //random intercept for mom if mother groomed or not
  
  real r_troop_mom_juve_j[TMJ]; //random intercept for juve on groomed or not
  real r_troop_mom_juve_m[TMJ]; //random intercept for juve if mother groomed or not
  
  
  // random effects standard deviations
  
  real<lower=0> sigma_troop_mom_juve_j;       //variation in juves: groomed or not
  real<lower=0> sigma_troop_mom_j;       //variation in moms: groomed or not 
  real<lower=0> sigma_troop_j;       //variation in troops: groomed or not
  
  real<lower=0> sigma_troop_mom_juve_m;      //variation in juves: mom groomed or not
  real<lower=0> sigma_troop_mom_m;      //variation in moms: mom groomed or not
  real<lower=0> sigma_troop_m;      //variation in troops: mom groomed or not

}

transformed parameters {
  
  
  //probabilities
  real p_j[N]; // conditional prob of juve being groomed 
  real p_m[N]; // conditional prob of mom grooming juve
  
  //estiamte probabilities for each observation
  for(i in 1:N){
    p_j[i] = beta[1] + beta[2]*age[i] + beta[3]*elo[i] + beta[4]*sex[i]+ r_troop_j[troop[i]] + r_troop_mom_j[troop_mom[i]] + r_troop_mom_juve_j[troop_mom_juve[i]];
    p_m[i] = beta[5] + beta[6]*age[i] + beta[7]*elo[i]+ beta[8]*sex[i] + r_troop_m[troop[i]] + r_troop_mom_m[troop_mom[i]] + r_troop_mom_juve_m[troop_mom_juve[i]]; 
  }
  
}


model {
  
  //priors
  
  // prior for all fixed-effects
  beta ~ normal(0, 1);   
  
  // model will estimate the prior for varying intercepts
  r_troop_j ~ normal(0, sigma_troop_j); 
  r_troop_m ~ normal(0, sigma_troop_m);
  r_troop_mom_j ~ normal(0, sigma_troop_mom_j);
  r_troop_mom_m ~ normal(0, sigma_troop_mom_m);
  r_troop_mom_juve_j ~ normal(0, sigma_troop_mom_juve_j);
  r_troop_mom_juve_m ~ normal(0, sigma_troop_mom_juve_m);
  
  // priors for the variation in the intercepts
  sigma_troop_mom_juve_j ~ normal(0,1);
  sigma_troop_mom_j ~ normal(0,1);
  sigma_troop_j ~ normal(0,1);
  sigma_troop_mom_juve_m ~ normal(0,1);
  sigma_troop_mom_m ~ normal(0,1);
  sigma_troop_m ~ normal(0,1);
  
  
  //likelihood
  
  //for each data point
  for (i in 1:N){
    
    if(gj[i] == 0){         //juve is not groomed
      
      0 ~ bernoulli_logit(p_j[i]);
    
    } else {               //juve is groomed

      1 ~ bernoulli_logit(p_j[i]);
      
      if (mg[i]==0){       //juve is not groomed by mom
    
        0 ~ bernoulli_logit(p_m[i]);
    
      } else {             //juve is groomed by mom
    
        1 ~ bernoulli_logit(p_m[i]);
    
      }
    }
  }
}



generated quantities {
  
  vector[N] y_pred_groom;
  for (i in 1:N) {
    y_pred_groom[i] = bernoulli_logit_rng(beta[1] + beta[2]*age[i] + beta[3]*elo[i] + beta[4]*sex[i]+ r_troop_j[troop[i]] + r_troop_mom_j[troop_mom[i]] + r_troop_mom_juve_j[troop_mom_juve[i]]); 
 
    }
    
  vector[N] y_pred_mom;
  for (i in 1:N) {
     y_pred_mom[i] = bernoulli_logit_rng(beta[5] + beta[6]*age[i] + beta[7]*elo[i] + beta[8]*sex[i]+ r_troop_m[troop[i]] + r_troop_mom_m[troop_mom[i]] + r_troop_mom_juve_m[troop_mom_juve[i]]);
 
    }
    
  vector[N] y_pred_groom_noRE;
  for (i in 1:N) {
   y_pred_groom_noRE[i] = bernoulli_logit_rng(beta[1] + beta[2]*age[i] + beta[3]*elo[i]+ beta[4]*sex[i]); 
 
    }
    
  vector[N] y_pred_mom_noRE;
 for (i in 1:N) {
  y_pred_mom_noRE[i] = bernoulli_logit_rng(beta[5] + beta[6]*age[i] + beta[7]*elo[i]+ beta[8]*sex[i]);
 
   }
  
  // //values to be generated
  // vector[500] y_groomed;
  // vector[500] y_groomed_by_mom;
  // 
  // //use every observation in the data
  // for (n in 1:500) {
  //   
  //   //generate a value for the observation: was the juve groomed?
  //   y_groomed[n] = bernoulli_logit_rng(p_j[n]);
  //   
  //   if(y_groomed[n]==0){
  //     
  //     //if not then the mom could not have groomed!
  //     y_groomed_by_mom[n]=0;
  //     
  //   } else {
  //     
  //     //if yes: did the mom groom the juve?
  //     y_groomed_by_mom[n] = bernoulli_logit_rng(p_m[n]);
  //   }
  //   
  // }
  // 
  
}
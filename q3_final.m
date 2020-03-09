%% Corrected EM question
%n = 2;
%% EM Algorithm
clear all;clc;
% Produce data sets from two dimensional gaussian models
n = 2;
mu(1,:) = [5.5; 2]; Sigma(:,:,1) = [1 0.5;0.5 1];
mu(2,:) = [5; 6];Sigma(:,:,2) = [1.2 0.6;0.6 1.2];
mu(3,:) = [4.5; 7];Sigma(:,:,3) = [1 0.1;0.1 1];
mu(4,:) = [3; 3];Sigma(:,:,4) = [0.6 0.2;0.2 0.6];

% use gmdistribution 
gm = gmdistribution(mu,Sigma);

% Create data from this model 
rng('default');
%X_10 = random(gm,10);
%X_100 = random(gm,100);
%X_1000 = random(gm, 1000);
% scatter(X_1000(:,1),X_1000(:,2));hold on;
for M = 1:100
X_1000 = random(gm, 1000);    
% % Start Iterations for M = 100
for b = 1:10
% % Select training and validating sets
    p = rand(1,1);
    x_train = datasample(X_1000,round(p*1000));
    x_valid = datasample(X_1000,round((1-p)*1000));
    for m = 1:6
         options = statset('Display','final','MaxIter',1500); 
         try
            GMM_Model{m} = fitgmdist(x_train,m,'CovarianceType','full','Start','plus','Options',options);
            GMM_Valid{m} = fitgmdist(x_valid,m,'CovarianceType','full','Start','plus','Options',options);
            likelihood(m) = GMM_Model{1,m}.NegativeLogLikelihood;
            iterations(m) = GMM_Model{1,m}.NumIterations;
            Bayesian_Inference(m) = GMM_Model{1,m}.BIC;
            Akaike_Inference(m) = GMM_Model{1,m}.AIC;
            performance(m) = GMM_Valid{1,m}.NegativeLogLikelihood;
            iterations_perf(m) = GMM_Valid{1,m}.NumIterations;
         catch exception
            disp('there was an error fitting the gaussian model');
            error = exception.message;
         end
    end 
    [maxval,index] = max(Akaike_Inference); 
%    [maxval_perf,index_perf] = min(performance);
    
    bestfit(b) = index;
%    bestval(b) = index_perf; 
    
% get performance of the model by using validation set
end     
%% find maxm of GMM from above cases
Bestfit_Data(M,:) = bestfit;
%Bestval_Data(M,:) = bestval;
end
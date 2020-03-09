clear all;clc;
% Produce data sets from two dimensional gaussian models
n = 2;
% mu = [-4 0 5 0;0 0 0 0];
% Sigma(:,:,1) = [1 0.5;0.5 1];
% Sigma(:,:,2) = [1.2 0.6;0.6 1.2];
% Sigma(:,:,3) = [1 0.1;0.1 1];
% Sigma(:,:,4) = [0.6 0.2;0.2 0.6];

mu = [-6 0 6 1;0 0 0 3];
Sigma(:,:,1) = [3 1;1 20];
Sigma(:,:,2) = [7 1;1 2];
Sigma(:,:,3) = [4 1;1 16];
Sigma(:,:,4) = [3 1;1 20];

alpha = [0.25,0.25,0.25,0.25];

X_1000 = randGMM(1000,alpha,mu, Sigma);


%% Model fit
for M =1:100
%performance = zeros(6,1);
performance_array = zeros(6,10000);
for B = 1:10 
    x_train = datasample(X_1000,10000,2);
    x_valid = datasample(X_1000,10000,2);
    for m = 1:6
         options = statset('Display','final','MaxIter',1500); 
         try
            GMM_Model{m} = fitgmdist(x_train',m,'CovarianceType','full','RegularizationValue',0.1,'Start','plus','Options',options);
             alpha_eval = GMM_Model{1,m}.ComponentProportion';
             mu_eval = GMM_Model{1,m}.mu';
             sigma_eval = GMM_Model{1,m}.Sigma;
             % get performance of the model by using validation set
             performance_array(m,:) =  performance_array(m,:) + evalGMM(x_valid,alpha_eval,mu_eval,sigma_eval); 
          catch exception
             disp('there was an error fitting the gaussian model');
             error = exception.message;
         end
    end 
%      performance =  performance + sum(performance_array,2);
end
[maxval,model_selected] = max(sum(performance_array,2)./10000);
Model_selected(M) = model_selected;
end
X = 1:6;
for i = 1:6
    Y(i) = sum(Model_selected==i);
end
bar(X,Y)
%% EM Algorithm
clear all;clc;
% Produce data sets from two dimensional gaussian models
n = 2;
% mu = [-4 0 5 0;0 0 0 0];
% Sigma(:,:,1) = [1 0.5;0.5 1];
% Sigma(:,:,2) = [1.2 0.6;0.6 1.2];
% Sigma(:,:,3) = [1 0.1;0.1 1];
% Sigma(:,:,4) = [0.6 0.2;0.2 0.6];

mu = [-10 0 10 0;0 0 0 0];
Sigma(:,:,1) = [3 1;1 20];
Sigma(:,:,2) = [7 1;1 2];
Sigma(:,:,3) = [4 1;1 16];
Sigma(:,:,4) = [3 1;1 20];
alpha = [0.1,0.6,0.1,0.2];

% Create data from this model 

%X_10 = random(gm,10);
%X_100 = random(gm,100);
%X_1000 = random(gm, 1000);
% scatter(X_1000(:,1),X_1000(:,2));hold on;
selected_model_X_10 = [];
for M = 1:100
X_10 = randGMM(10,alpha,mu, Sigma);
% % Start Iterations for M = 100
for b = 1:10
% % Select training and validating sets
    
    x_train = datasample(X_10,50,2);
    x_valid = datasample(X_10,50,2);
    performance_array = zeros(1,6);
    for m = 1:6
         options = statset('Display','final','MaxIter',1500); 
         try
            GMM_Model{m} = fitgmdist(x_train,m,'CovarianceType','full','Start','plus','Options',options);
             alpha_eval = GMM_Model{1,m}.ComponentProportion';
             mu_eval = GMM_Model{1,m}.mu';
             sigma_eval = GMM_Model{1,m}.Sigma;
             % get performance of the model by using validation set
             performance_array(m) =  performance_array(m) + evalGMM(x_valid,alpha_eval,mu_eval,sigma_eval); 
         catch exception
            disp('there was an error fitting the gaussian model');
            error = exception.message;
         end
    end 
end 
performance_array = performance_array./10;
%find maxm of GMM from above cases
[performance_value, Model_order] = max(performance_array);
selected_model_X_10 = [selected_model_X_10;Model_order];
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end
%% eval GMM
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%% eval Gaussian 
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%% rand Gaussian
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
%% Generation of  data 
clear all; close all; clc;

n = 2;      % number of feature dimensions
N_10 = 10;
N_100 = 100;    % number of iid samples for training
N_1000 = 1000;
N_10K = 10000;
% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 

% Class priors for class 0 and 1 respectively
p = [0.7,0.3]; 

% Generating true class labels and training data
label_10 = (rand(1,N_10) >= p(1))';
Nc_10 = [length(find(label_10==0)),length(find(label_10==1))];
label_100 = (rand(1,N_100) >= p(1))';
Nc_100 = [length(find(label_100==0)),length(find(label_100==1))];
label_1000 = (rand(1,N_1000) >= p(1))';
Nc_1000 = [length(find(label_1000==0)),length(find(label_1000==1))];
% Generating true class labels and validating data
label_10K = (rand(1,N_10K) >= p(1))';
Nc_10K = [length(find(label_10K==0)),length(find(label_10K==1))];

x_10 = zeros(N_10,n);
x_100 = zeros(N_100,n);
x_1000 = zeros(N_1000,n);
x_10K = zeros(N_10K,n);

for L = 0:1
    x_10(label_10==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_10(L+1));
    x_100(label_100==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_100(L+1));
    x_1000(label_1000==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_1000(L+1));
    x_10K(label_10K==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_10K(L+1));
end

% Visualizing the data
figure(1), clf,
plot(x_10K(label_10K==0,1),x_10K(label_10K==0,2),'o'), hold on,
plot(x_10K(label_10K==1,1),x_10K(label_10K==1,2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 


%% Minimum P(error) classifier
%gamma = 0.7/0.3;
discriminantScore = log(evalGaussian(x_10K,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x_10K,mu(:,1),Sigma(:,:,1)));% - log(gamma);
p_detection = zeros(10000,1);
p_false_alarm = zeros(10000,1);
gammaArray = 0:0.5:250;

for i = 1:size(gammaArray,2)
  
    gamma = gammaArray(i);
    decision = (discriminantScore >= log(gamma));
    ind00 = find(decision==0 & label_10K'==0); p00 = length(ind00)/Nc_10K(1); % probability of true negative
    ind10 = find(decision==1 & label_10K'==0); p10 = length(ind10)/Nc_10K(1); % probability of false positive
    ind01 = find(decision==0 & label_10K'==1); p01 = length(ind01)/Nc_10K(2); % probability of false negative
    ind11 = find(decision==1 & label_10K'==1); p11 = length(ind11)/Nc_10K(2); % probability of true positive
    p_detection(i) = p11;
    p_false_alarm(i) = p10;
    
end

figure(2)
plot(p_false_alarm,p_detection,'b');hold on;

% Minimum error

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x_10K,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x_10K,mu(:,1),Sigma(:,:,1)))
decision = (discriminantScore >= log(gamma));

    ind00 = find(decision==0 & label_10K'==0); p00 = length(ind00)/Nc_10K(1);
    ind10 = find(decision==1 & label_10K'==0); p10 = length(ind10)/Nc_10K(1);
    ind01 = find(decision==0 & label_10K'==1); p01 = length(ind01)/Nc_10K(2); 
    ind11 = find(decision==1 & label_10K'==1); p11 = length(ind11)/Nc_10K(2); 
    p_error = [p10,p01]*Nc_10K'/N_10K;

figure(2)
scatter(p10,p11,'r*');
title('ROC Curve and marking the minimum error possible point')
xlabel('Probability of false detection'),ylabel('Probability of detection');


%% Plot Decisions
figure(3), % class 0 circle, class 1 +, correct green, incorrect red
plot(x_10K(ind00,1),x_10K(ind00,2),'og'); hold on,
plot(x_10K(ind10,1),x_10K(ind10,2),'or'); hold on,
plot(x_10K(ind01,1),x_10K(ind01,2),'+r'); hold on,
plot(x_10K(ind11,1),x_10K(ind11,2),'+g'); hold on,
axis equal,

%% Plot Boundaries
horizontalGrid = linspace(floor(min(x_10K(:,1))),ceil(max(x_10K(:,1))),101);
verticalGrid = linspace(floor(min(x_10K(:,2))),ceil(max(x_10K(:,2))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian(reshape([h(:)';v(:)']',91*101,2),mu(:,2),Sigma(:,:,2)))-log(evalGaussian(reshape([h(:)';v(:)']',91*101,2),mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(3), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]);
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 
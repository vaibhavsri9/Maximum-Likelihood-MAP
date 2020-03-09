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
discriminantScore = log(evalGaussian(x_10K,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x_10K,mu(:,1),Sigma(:,:,1)));
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

%% Plot Boundaries (MER)
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
% clear variables except data and true labels
clearvars -except x* & label* & N* & p*

%% Logistic Linear Regression
% input N (in this case I have selected 1000)
N = 10;
n = 2;
x = [ones(N, 1) x_10];
x_quad = [ones(N, 1) x_10(:,1) x_10(:,2) x_10(:,1).^2 x_10(:,1).*x_10(:,2) x_10(:,2).^2];
label_10=double(label_10);
initial_theta = zeros(n+1, 1);
initial_theta2 = zeros(6, 1);
% To display optimization over iterations
options = optimset('Display','iter','PlotFcns',@optimplotfval);
% Compute minimum
[theta2,cost2,exitflag,output] = fminsearch(@(t)(cost_func(t, x, label_10, N)), initial_theta,options);
[theta3,cost3,exitflag,output] = fminsearch(@(t)(cost_func(t, x_quad, label_10, N)), initial_theta2,options);

% Points for decision boundary
plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];
plot_x2(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x1 + theta2(1));% fminsearch

% Conditions where real solution for quadratic exist
P = (theta3(5)^2) + (4*theta3(6)*theta3(4));
Q = (2*theta3(5)*theta3(3))-(4*theta3(6)*theta3(2));
R = (theta3(3)^2)-(4*theta3(6)*theta3(1));
plot_x1_quad = -0.2:0.1:max(plot_x1(:,2))+2;
S = P.*(plot_x1_quad.^2) + (Q.*plot_x1_quad) + R; %Confirmation of real
%roots
A = theta3(6);
B = (theta3(5).*plot_x1_quad) + theta3(3);
if S > 0
    plot_x2_quad1(1,:) = (-B + sqrt(S))./(2*A);
    plot_x2_quad2(1,:) = (-B - sqrt(S))./(2*A);
else 
    plot_x1_quad = 1:0.1:8;
    S = P.*(plot_x1_quad.^2) + (Q.*plot_x1_quad) + R;
    plot_x2_quad3(1,:) = (-B + sqrt(S))./(2*A);
    plot_x2_quad4(1,:) = (-B - sqrt(S))./(2*A);
end

%Plotting decision boundary on data
figure(4)
plot(x_10(label_10==0,1),x_10(label_10==0,2),'o',x_10(label_10==1,1),x_10(label_10==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;
plot(plot_x1, plot_x2(2,:));  hold on;
plot(plot_x1_quad, plot_x2_quad1(1,:),'r-');hold on;
plot(plot_x1_quad, plot_x2_quad2(1,:),'r-');hold on;
axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', 'Classifier (fminsearch)','Classifier (Quadratic)');

%% Testing the designed classifier with validating data  
% For linear
coeff(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1);
i = 1;
    if coeff(i,1) >= 0
        decision(:,i) = (coeff(i,1).*x_10K(:,1) + coeff(i,2)) > x_10K(:,2);
    else 
        decision(:,i) = (coeff(i,1).*x_10K(:,1) + coeff(i,2)) < x_10K(:,2);
    end
error1 = plot_test_data(decision(:,1), label_10K, Nc_10K, p, 5, x_10K, plot_x1, plot_x2(2,:));
title('Test Data Classification (Linear)');
fprintf('Total error (linear-fminsearch): %.2f%%\n',error1);

% For Quadratic
for m = 1:10000   
h = [1 x_10K(m,1) x_10K(m,2) x_10K(m,1)^2 x_10K(m,1)*x_10K(m,2) x_10K(m,2)^2];
    if x_10K(m,1) > 0 
       decision_quad(m,1) = h*theta3 > 0;
    else
       decision_quad(m,1) = false; 
    end
end

error2 = plot_test_data_quad(decision_quad(:,1), label_10K, Nc_10K, p, 6, x_10K, plot_x1_quad, plot_x2_quad1, plot_x1_quad, plot_x2_quad2);
title('Test Data Classification (Quadratic)');
fprintf('Total error (Quadratic-fminsearch): %.2f%%\n',error2);
%% Define Cost function which for which the parameter needs to be minimised
function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end
%% Define error function 
function error = plot_test_data(decision, label, Nc, p, fig, x, plot_x1, plot_x2)
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(plot_x1, plot_x2);
    axis([plot_x1(1), plot_x1(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Incorrect Decisions','Class 1 Incorrect Decisions','Class 1 Correct Decisions','Classifier');
end
%% Define error function for quadratic
function error_quad = plot_test_data_quad(decision, label, Nc, p, fig, x, plot_x1, plot_x2, plot_x3, plot_x4)
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % true positive
    error_quad = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(plot_x1, plot_x2, '-m'); hold on,
    plot(plot_x3, plot_x4, '-m'); 
    axis([-10 10 -10 10])
    legend('Class 0 Correct Decisions','Class 0 Incorrect Decisions','Class 1 Incorrect Decisions','Class 1 Correct Decisions','Classifier');
end

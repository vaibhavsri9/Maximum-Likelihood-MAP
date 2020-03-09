%% Parameter estimation
clc; clear all; 
% Initializing with roots between (1,-1) with roots 0.2,0.6 and -0.5
N = 10;
a = 1;
b = -0.3;
c = -0.28;
d = 0.06;
Sigma = 0.05;
gammaArray = 10.^[-5:1:5];
%gammadefault = 10^3;
w_true = [a,b,c,d];
% generating iid samples 
% where is this being used
%% Calculations
for q = 1:size(gammaArray,2)
    gamma = gammaArray(q);
    for k= 1:100
        x = -1 + 2*rand(1,N);
        V = mvnrnd(0,Sigma,N);
        z = [x(1,:).^3;x(1,:).^2;x(1,:);ones(1,N)];
        y = (w_true*z) + V';
% Computing product of z*z' 
        for i = 1:N
            zzT(:,:,i) = z(:,i)*z(:,i)';
        end
%% Calculate ML estimate (Review along which dimension does the addition take place)
        pr_ML = inv(sum(zzT,3));
        qr = sum(y.*z,2);
        w_ML = pr_ML*qr;
%% MAP estimate 
         pr_MAP = inv(sum(zzT,3) + ((Sigma^2)/(gamma^2)).*eye(4));
         w_MAP = pr_MAP*qr;    
         L2(q,k) = (norm(w_true'-w_MAP,2))^2;
        end
end


boxplot(L2','DataLim',[0,3],'Labels',gammaArray);
title('(L_2)^2 over Gamma')


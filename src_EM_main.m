%% MetroAssignment
% -- Lijun Sun -- %
% -- last modified: April 20, 2015 -- %
% This script implements EM algorithm to infer network attributes and
% passenger choice behavior parameters
%
% This function uses repmat heavily (repmat has a better performance than bsxfun on my PC with matlab 2014a)
% http://stackoverflow.com/questions/28722723/matlab-bsxfun-no-longer-faster-than-repmat
% If using a previous version of matlab, you may revise the repmat functions to bsxfun(@plus,@minus,@times,@rdivide,....)
% Please consider to cite the following article when using this code
% Sun, L., Lu, Y., Jin, J.G., Lee, D.-H., Axhausen, K.W., 2015. An integrated Bayesian approach for passenger flow assignment in metro networks. Transportation Research Part C: Emerging Technologies 52, 116-131.


clear all; clc; close all;
% global transfer_num;
% the following data are cellarrays with size = OD pairs
% each element is for one OD pair
global transfer_link; % logical matrix indicating whether a transfer link is used on a route
global travel_time; % travel time observations
global route; % logical matrix indicating whether a link (both in-vehicle and transfer) is used on a route
global Ezk; % responsibility; a matrix showing the probability of a travel time observation comes from a certain route

global num; % total number of OD pairs
global len; % number of travel time observations, an array with size = OD pairs
global varwait; % sigma_y^2; variance of waiting time (y)
global mu_cost; % link cost
global nlinks; % number of links

sss = 5; % hourly data; which hour

load(strcat('data/data_mu_cost_big_',num2str(sss)));
load(strcat('data/data_travel_time_big_',num2str(sss)));
load(strcat('data/data_route_big_',num2str(sss)));
load(strcat('data/data_transfer_link_big_',num2str(sss)));
nlinks = length(mu_cost);

for i = 1:length(route)
    route{i} = logical(route{i});
    transfer_link{i} = logical(transfer_link{i});
end

%% route/ check data consistency
num = length(route);
check = zeros(num,1);
for i = 1:num
    if size(transfer_link{i}) == size(route{i})
        check(i) = 1;
    end
end
sum(check)
clear check;

% initialize number of potential routes on each OD pair
len = zeros(length(route),1);
for i = 1:length(route)
    len(i) = size(route{i},1);
end
varwait = 1.5;

% maximum number of observations used is 100; extend travel times to a matrix given potential routes
for i = 1:length(travel_time)
    if length(travel_time{i})>=100
        travel_time{i} = randsample(travel_time{i},100);
    end
    travel_time{i} = repmat(travel_time{i},[1,len(i)]);
end

%% with alpha
global temp_mu_old;
global temp_alpha_old;
global temp_theta_old;
global temp_waiting_old;

load tt; % initial parameters
temp_mu_old = tt(1:107);
temp_alpha_old = tt(108:109);
temp_theta_old = tt(110:111);
temp_waiting_old = tt(112);

% set options for optimization in M-step, given the version of matlab
% matlab 2013 and above
% options = optimoptions(@fminunc,'display','iter','algorithm','quasi-newton','maxiter',10);
% matlab 2012
options = optimset('display','iter','largescale','on','maxiter',10);

Ezk = cell(num,1);
final_res = zeros(100,112);
final_value = zeros(100,1);

%%
% the EM algorithm is run for 100 iterations (doing convergence check)
for iter = 1:50
    % find EZK
    % the latend variable is coded as a logical matrix
    for i = 1:num
        % E-step: update responsibility
        if len(i) > 1 % more than one route
            
            % get links used
            mu_mat = route{i}*temp_mu_old' + temp_waiting_old;
            transfer_mat = transfer_link{i}*temp_mu_old';
            sigma2_mat = (route{i}-transfer_link{i})*((temp_alpha_old(1)*temp_mu_old').^2) +...
                transfer_link{i}*((temp_alpha_old(2)*temp_mu_old').^2) + varwait;
            % MNL model
            choice = exp(temp_theta_old(1)*(mu_mat-temp_waiting_old-transfer_mat)+temp_theta_old(2)*transfer_mat);
            wchoice = choice./sum(choice);
            mu_mat = repmat(mu_mat', [size(travel_time{i},1),1]);
            sigma2_mat = repmat(sigma2_mat', [size(travel_time{i},1),1]);
            prob = exp(-0.5*(travel_time{i} - mu_mat).^2 ./sigma2_mat) ./ sqrt(2*pi*sigma2_mat);
            Ezkt = repmat(wchoice',size(travel_time{i},1),1);
            Ezkt = Ezkt.*prob;
            Ezk{i} = Ezkt./repmat(sum(Ezkt,2),1,len(i));
        else % only one possible route, probability of choosing it is 1
            Ezk{i} = ones(size(travel_time{i},1),1);
        end
    end
    disp(iter);
    x = [temp_mu_old, temp_alpha_old, temp_theta_old, temp_waiting_old];
    % M-step: calculating new parameters
    tic;
    [res,fv] = fminunc(@src_EM_likelihood_alpha,x,options);
    toc;
    final_res(iter,:) = res;
    final_value(iter) = fv;
    % update parameters
    temp_mu_old = res(1:107);
    temp_alpha_old = abs(res(108:109));
    temp_theta_old = res(110:111);
    temp_waiting_old = res(112);
    close all;
end

%% test the performance of log-likelihood function
x = [temp_mu_old, temp_alpha_old, temp_theta_old, temp_waiting_old];
profile on;
for i = 1:10
    src_EM_likelihood_alpha(x)
end
profile viewer;

%% save results in a .mat file

save(strcat('res_',num2str(sss),'.mat'),'final_res','final_value');
function lglik = src_EM_likelihood_alpha(vari)
% -- Lijun Sun -- %
% -- last modified: April 20, 2015 -- %
% src_EM_likelihood_alpha calculating the log-likelihood of parameters given observting (travel time)
%
% vari is a vector of all parameters
% This function uses repmat heavily (repmat has a better performance than bsxfun on my PC with matlab 2014a)
% http://stackoverflow.com/questions/28722723/matlab-bsxfun-no-longer-faster-than-repmat
% if using a previous version of matlab, you may revise the repmat functions to bsxfun(@plus,@minus,@times,@rdivide,....)

% Please consider to cite the following article when using this code
% Sun, L., Lu, Y., Jin, J.G., Lee, D.-H., Axhausen, K.W., 2015. An integrated Bayesian approach for passenger flow assignment in metro networks. Transportation Research Part C: Emerging Technologies 52, 116-131.

% global transfer_num;
% the following data are cellarrays with size = OD pairs
% each element is for one OD pair
global transfer_link; % logical matrix indicating whether a transfer link is used on a route
global travel_time; % travel time observations
global route; % logical matrix indicating whether a link (both in-vehicle and transfer) is used on a route

global num; % total number of OD pairs
global varwait; % sigma_y^2; variance of waiting time (y)

global Ezk; % responsibility; a matrix showing the probability of a travel time observation comes from a certain route

temp_mu_new = vari(1:107)'; % temp_mu_new is reshaped as a vector
temp_alpha_new = vari(108:109);
temp_theta_new = vari(110:111);
temp_waiting_new = vari(112);

lglik = 0;
for i = 1: num
    % get links used
    mu_mat_new = route{i}*temp_mu_new;
    transfer_mat = transfer_link{i}*temp_mu_new;
    sigma2_mat_new = (route{i} - transfer_link{i})*((temp_alpha_new(1)*temp_mu_new).^2) + transfer_link{i}*((temp_alpha_new(2)*temp_mu_new).^2) + varwait;
    time2 = -0.5*log(sigma2_mat_new);
    % MNL model
    dschoice1 = temp_theta_new(1)*(mu_mat_new-transfer_mat)+temp_theta_new(2)*transfer_mat;
    %dschoice2 = -log(sum(exp(dschoice1)));
    dschoice2 = -logsumexp(dschoice1,1)
    % using bsxfun (seems to be slower on my PC)
    % time1 = bsxfun(@rdivide,-0.5*bsxfun(@minus,travel_time{i},mu_mat_new').^2,sigma2_mat_new');
    % using repmat 
    mu_mat_new = repmat(mu_mat_new', [size(travel_time{i},1),1]) + temp_waiting_new;
    sigma2_mat_new = repmat(sigma2_mat_new', [size(travel_time{i},1),1]);
    time1 = -0.5*(travel_time{i} - mu_mat_new).^2 ./sigma2_mat_new;
    
    tot = sum(Ezk{i});
    lglik = lglik - (tot*(time2+dschoice1+dschoice2) + sum(sum(Ezk{i}.*time1)) );
end
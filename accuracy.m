function [p_acc,a_acc,pair_acc] = accuracy(c_nu,mapping,xData,flag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was used for computing accuracy on CASABLANCA dataset
% c_nu is the probability output from IBP model
% mapping denotes the actual ordering of latent factors
% xData contains the data and groundtruth for evaluation
% flag = 1 for computing precision and recall curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a_nu=cell2mat(reshape(c_nu',numel(c_nu),1));
a_nu = a_nu(:,mapping);
% a_nu = a_nu(:,1:(xData.num_persons + xData.num_actions));

a_gt=cell2mat(reshape(xData.xTopicList',numel(xData.xTopicList),1));

idx = find(max(a_gt(:,1:xData.num_persons),[],2) > 0);
per_gt = zeros(size(a_gt,1),1);
[~,per_gt(idx)] = max(a_gt(idx,1:xData.num_persons),[],2);

idx = find(max(a_gt(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) > 0);
act_gt = zeros(size(a_gt,1),1);
[~,act_gt(idx)] = max(a_gt(idx,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2);

% Computing precision and recall plot for both background and non-background factors
if(flag)
    p_nbg_r = zeros(100,1);
    p_bg_r = zeros(100,1);
    a_nbg_r = zeros(100,1);
    a_bg_r = zeros(100,1);
    p_nbg_p = zeros(100,1);
    p_bg_p = zeros(100,1);
    a_nbg_p = zeros(100,1);
    a_bg_p = zeros(100,1);
    thres_idx = 1;
    for thres = 0.01:0.01:1
        idx = find(max(a_nu(:,1:xData.num_persons),[],2) > thres); % 0.2
        per_assign = zeros(size(a_nu,1),1);
        [~,per_assign(idx)] = max(a_nu(idx,1:xData.num_persons),[],2);
        
        p_nbg_r(thres_idx) = sum(per_assign(per_gt ~= 0) == per_gt(per_gt ~= 0))/sum(per_gt ~= 0);
        p_bg_r(thres_idx) = sum(per_assign(per_gt == 0) == per_gt(per_gt == 0))/sum(per_gt == 0);
        
        p_nbg_p(thres_idx) = sum(per_assign(per_gt ~= 0) == per_gt(per_gt ~= 0))/sum(per_assign ~= 0);
        p_bg_p(thres_idx) = sum(per_assign(per_gt == 0) == per_gt(per_gt == 0))/sum(per_assign == 0);
        
        idx = find(max(a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) > thres);
        act_assign = zeros(size(a_nu,1),1);
        [~,act_assign(idx)] = max(a_nu(idx,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2);
        
        a_nbg_r(thres_idx) = sum(act_assign(act_gt ~= 0) == act_gt(act_gt ~= 0))/sum(act_gt ~= 0);
        a_bg_r(thres_idx) = sum(act_assign(act_gt == 0) == act_gt(act_gt == 0))/sum(act_gt == 0);
        
        a_nbg_p(thres_idx) = sum(act_assign(act_gt ~= 0) == act_gt(act_gt ~= 0))/sum(act_assign ~= 0);
        a_bg_p(thres_idx) = sum(act_assign(act_gt == 0) == act_gt(act_gt == 0))/sum(act_assign == 0);
        
        thres_idx = thres_idx + 1;
    end
end

% This is based on PR curve. So threshold values will vary with datasets.
% threshold 0.2 is use to detect non-background subject track from background tracks
idx = find(max(a_nu(:,1:xData.num_persons),[],2) > 0.2); 
per_assign = zeros(size(a_nu,1),1);
[~,per_assign(idx)] = max(a_nu(idx,1:xData.num_persons),[],2);

% threshold 0.02 is use to detect non-background actions track from background tracks
idx = find(max(a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) > 0.02);
act_assign = zeros(size(a_nu,1),1);
[~,act_assign(idx)] = max(a_nu(idx,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2);

if(flag)
    [cp,op] = confusionmat(per_gt,per_assign);
    [ca,oa] = confusionmat(act_gt,act_assign);
    
    idx = find(max(a_nu(:,1:xData.num_persons),[],2) <= 0.2);
    Y = zeros(size(a_gt,1),1);
    Y(idx) = mean(a_nu(idx,1+xData.num_actions+xData.num_persons:end),2);
    Y = [Y a_nu(:,1:xData.num_persons)];
    presults = evaluate(Y, per_gt+1);
    
    idx = find(max(a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) <= 0.02);
    Y = zeros(size(a_gt,1),1);
    Y(idx) = mean(a_nu(idx,1+xData.num_actions+xData.num_persons:end),2);
    Y = [Y a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions)];
    aresults = evaluate(Y, act_gt+1);
    
    save('roc.mat','p_nbg_r','p_bg_r','a_nbg_r','a_bg_r','p_nbg_p','p_bg_p','a_nbg_p','a_bg_p','cp','op','ca','oa','presults','aresults');
end

% Compute individual accuracy of person and actions and print them
for i = 0:xData.num_persons
    idx1 = per_gt == i;
    p_acc = sum(per_assign(idx1) == per_gt(idx1))/sum(idx1);
    fprintf('%d \t %1.4f\n',i,p_acc);
end
for i = 0:xData.num_actions
    idx2 = act_gt == i;
    a_acc = sum(act_assign(idx2) == act_gt(idx2))/sum(idx2);
    fprintf('%d \t %1.4f\n',i,a_acc);
end

% Considers only non-background tracks for pairwise accuracy computations
temp = per_gt ~= 0 & act_gt ~= 0;

% Compute pairwise accuracy
pair_acc = sum(per_assign(temp) == per_gt(temp) & act_assign(temp) == act_gt(temp))/sum(temp);

% Compute total accuracy of person and actions
p_acc = sum(per_assign == per_gt)/length(per_gt);
a_acc = sum(act_assign == act_gt)/length(act_gt);
end
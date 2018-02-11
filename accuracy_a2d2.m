function [p_acc,a_acc,pair_acc,iou,iou1] = accuracy_a2d2(c_nu,rev_ordering,xData,flag,D1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was used for computing accuracy on A2D dataset
% c_nu is the probability output from IBP model
% mapping denotes the actual ordering of latent factors
% xData contains the data and groundtruth for evaluation
% flag = 1 for computing precision and recall curve
% D1 is the dimension of subject features
% iou, iou1 is the IOU measured using two different metrics. Please see the
% code below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iou = 0;
iou1 = 0;
a_nu=cell2mat(reshape(c_nu',numel(c_nu),1));
a_nu = a_nu(:,rev_ordering);

per_gt = xData.trklabel(:,1);
act_gt = xData.trklabel(:,2);

if(flag)
    % assignment
    p_nbg = zeros(100,1);
    p_bg = zeros(100,1);
    a_nbg = zeros(100,1);
    a_bg = zeros(100,1);
    thres_idx = 1;
    for thres = 0.01:0.01:1
        % person
        idx = find(max(a_nu(:,1:xData.num_persons),[],2) > thres);
        per_assign = zeros(size(a_nu,1),1);
        [~,per_assign(idx)] = max(a_nu(idx,1:xData.num_persons),[],2);
        
        p_nbg(thres_idx) = sum(per_assign(per_gt ~= 0) == per_gt(per_gt ~= 0))/sum(per_gt ~= 0);
        p_bg(thres_idx) = sum(per_assign(per_gt == 0) == per_gt(per_gt == 0))/sum(per_gt == 0);
        
        % actions
        idx = find(max(a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) > thres);
        act_assign = zeros(size(a_nu,1),1);
        [~,act_assign(idx)] = max(a_nu(idx,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2);
        
        a_nbg(thres_idx) = sum(act_assign(act_gt ~= 0) == act_gt(act_gt ~= 0))/sum(act_gt ~= 0);
        a_bg(thres_idx) = sum(act_assign(act_gt == 0) == act_gt(act_gt == 0))/sum(act_gt == 0);
        
        thres_idx = thres_idx + 1;
    end
    save(['roc_',num2str(D1),'.mat'],'p_nbg','p_bg','a_nbg','a_bg');
end

idx = find(max(a_nu(:,1:xData.num_persons),[],2) > 0.99);
per_assign = zeros(size(a_nu,1),1);
[~,per_assign(idx)] = max(a_nu(idx,1:xData.num_persons),[],2);

idx = find(max(a_nu(:,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2) > 0.99);
act_assign = zeros(size(a_nu,1),1);
[~,act_assign(idx)] = max(a_nu(idx,xData.num_persons+1:xData.num_persons+xData.num_actions),[],2);

% mean average precision

% evaluation
for i = 1:xData.num_persons
    idx1 = per_gt == i;
    p_acc = sum(per_assign(idx1) == per_gt(idx1))/sum(idx1);
    [~,~,info] = vl_pr(2*idx1(xData.istest)-1,a_nu(xData.istest,i));
    p_ap(i) = info.ap;
    fprintf('%d \t %1.4f\t%1.4f\n',i,p_acc,p_ap(i));
end

% for background action
idx2 = act_gt == 0;
a_acc = sum(act_assign(idx2) == act_gt(idx2))/sum(idx2);
[~,~,info] = vl_pr(2*idx2(xData.istest)-1,max(a_nu(xData.istest,1+xData.num_persons+xData.num_actions:end),[],2));
a_ap(1) = info.ap;
fprintf('%d \t %1.4f\t%1.4f\n',0,a_acc,a_ap(1));

for i = 1:xData.num_actions
    idx2 = act_gt == i;
    a_acc = sum(act_assign(idx2) == act_gt(idx2))/sum(idx2);
    [~,~,info] = vl_pr(2*idx2(xData.istest)-1,a_nu(xData.istest,i+xData.num_persons));
    a_ap(i+1) = info.ap;
    fprintf('%d \t %1.4f\t%1.4f\n',i,a_acc,a_ap(i+1));
end

temp = per_gt ~= 0 & act_gt ~= 0;
pair_acc = sum(per_assign(temp) == per_gt(temp) & act_assign(temp) == act_gt(temp))/sum(temp);

% Test accuracy
p_acc = sum(per_assign(xData.istest) == per_gt(xData.istest))/sum(xData.istest);
a_acc = sum(act_assign(xData.istest) == act_gt(xData.istest))/sum(xData.istest);
fprintf('Test Accuracy\n');
fprintf('%1.4f\t%1.4f\t%1.4f\t%1.4f\n',p_acc, a_acc, mean(p_ap),mean(a_ap));

p_acc = sum(per_assign == per_gt)/length(per_gt);
a_acc = sum(act_assign == act_gt)/length(act_gt);
[iou,indx] = cellfun(@f_iou, xData.vidlabel, xData.iou, mat2cell(a_nu,10*ones(xData.imgLength,1)),'UniformOutput',false);
if(flag)
    save(['final_indx_',num2str(D1),'.mat'],'indx','iou');
end
iou = mean(cat(1,iou{:}));

iou1 = cellfun(@f_iou_t, xData.vidlabel, xData.iou, mat2cell(a_nu,10*ones(xData.imgLength,1)),'UniformOutput',false);
iou1 = mean(cat(1,iou1{:}));
end

function x = f_iou_t(gt,iou,nu)
x = [];
indx = [];
if(isempty(gt))
    return;
end
x = zeros(size(gt,1),1);
score = zeros(1,size(gt,1));
nu = [ones(10,1) nu];
% Selecting the best bounding box based on sum prob of correlated concepts
for i=1:size(gt,1)
    [score(i),~] = max(sum(nu(:,gt(i,:)),2));
end
[~,scidx] = sort(score,'descend');
for i=scidx
    confsc = sum(nu(:,gt(i,:)),2);
    confsc(indx(indx > 0)) = 0;
    [~,idx] = max(confsc);
    x(i) = iou(idx,i);
    indx(i) = idx;
end
end

function [x,indx] = f_iou(gt,iou,nu)
x = [];
indx = [];
if(isempty(gt))
    return;
end
x = zeros(size(gt,1),1);
score = zeros(1,size(gt,1));
nu = [ones(10,1) nu];
% Selecting the best bounding box based on product of prob of correlated concepts
for i=1:size(gt,1)
    [score(i),~] = max(prod(nu(:,gt(i,:)),2));
end
[~,scidx] = sort(score,'descend');
for i=scidx
    confsc = prod(nu(:,gt(i,:)),2);
    confsc(indx(indx > 0)) = 0;
    [~,idx] = max(confsc);
    x(i) = iou(idx,i);
    indx(i) = idx;
end
end

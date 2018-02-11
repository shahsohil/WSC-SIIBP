function [learnedModel,p_acc,a_acc,c_acc] = wscsiibp(para,model,xData)

% Initialiazing parameters of variational model i.e., \tau, \nu, \phi, \sigma_k (Phi)
c_nu=model.c_nu;
c_tau=model.c_tau;
Phi=model.Phi;
phi=model.phi;

% Initialiazing the appearance and the noise variance for the IBP model
sigma_A1=para.sigmaA1;
sigma_n1=para.sigmaN1;
sigma_A2=para.sigmaA2;
sigma_n2=para.sigmaN2;

% Maximum number of latent features
K=para.K;

% regularizing(weighing) parameter for the constraints
C=para.C;

% Dimension of input features for subject and action concept
D1=para.D1;
D2=para.D2;

% Parameter of beta distribution 
alpha=para.alpha;

feature_idx = 1:K;
a_X=xData.allX;

fprintf('Iter\t Person Acc\t Action Acc\t Pairwise Acc\n');

for I = 1:para.MAX_ITERATIONS
    a_nu=cell2mat(reshape(c_nu',numel(c_nu),1));

%   As per the Algorithm 1 line 10, appearance and noise variances are updated after every T iterations (epochs).    
    if(mod(I,10) == 0)
        sigma_A1 = sqrt((D1*sum(Phi(1,1,:)) + trace(phi(:,1:D1)*phi(:,1:D1)'))/(K*D1));
        % For our experiment we find that updating \sigma_ns worsen the performance.        
%         sigma_n1 = sqrt((sum(sum(a_X(:,1:D1).*a_X(:,1:D1))) - 2*sum(sum((a_X(:,1:D1)*phi(:,1:D1)').*a_nu)) ...
%             + trace(a_nu*(phi(:,1:D1)*phi(:,1:D1)')*a_nu') - trace(a_nu*diag(diag(phi(:,1:D1)*phi(:,1:D1)'))*a_nu') ...
%             + D1*sum(a_nu*squeeze(Phi(1,1,:))) + sum(a_nu*diag(phi(:,1:D1)*phi(:,1:D1)')))/(D1*xData.allSegLength));
        sigma_A2 = sqrt((D2*sum(Phi(D1+1,D1+1,:)) + trace(phi(:,D1+1:end)*phi(:,D1+1:end)'))/(K*D2));
        sigma_n2 = sqrt((sum(sum(a_X(:,D1+1:end).*a_X(:,D1+1:end))) - 2*sum(sum((a_X(:,D1+1:end)*phi(:,D1+1:end)').*a_nu)) ...
            + trace(a_nu*(phi(:,D1+1:end)*phi(:,D1+1:end)')*a_nu') - trace(a_nu*diag(diag(phi(:,D1+1:end)*phi(:,D1+1:end)'))*a_nu') ...
            + D2*sum(a_nu*squeeze(Phi(D1+1,D1+1,:))) + sum(a_nu*diag(phi(:,D1+1:end)*phi(:,D1+1:end)')))/(D2*xData.allSegLength));
    end

%    Code for reordering the features. This is done only during the first 7 iterations    
    if I<8
        % Due to the structure of the learning algorithm of IBP model one requires the latent features to be re-ordered.
        temp_tau=reshape(c_tau',numel(c_tau),1);
        
        % reordering is decided based on the nu
        [tmp, feature_order] = sort(sum(a_nu), 'descend');
        
        % reordering all the features one after another
        for temp_i=1:length(temp_tau)
            if(~isempty(temp_tau{temp_i}))
                temp_tau{temp_i}= temp_tau{temp_i}(feature_order,:);
            end
        end
        a_tau = cell2mat(temp_tau);
        phi= phi(feature_order,:);
        Phi =Phi(:,:,feature_order);
        a_nu = a_nu(:,feature_order);
        
        feature_idx = feature_idx(feature_order);
        
        % storing the reverse ordering to retrieve back the original index
        rev_ordering = [];
        for r=1:length(feature_idx)
            rev_ordering = [rev_ordering find(feature_idx == r)];
        end
        
    else
%       Unfolding the total videos into \sum N_i        
        a_tau=cell2mat(reshape(c_tau',numel(c_tau),1));
    end   

    %   As per the Algorithm 1 line 6, variational model parameters \phi and \sigma_k is updated    
    for k = 1:K
        Phi(1:D1,1:D1,k) = (1/sigma_A1^2 + 1/sigma_n1^2 * sum(a_nu(:,k)))^-1 * eye(D1);
        phi(k,1:D1) = 1/sigma_n1^2 * a_nu(:,k)'*(a_X(:,1:D1)-a_nu*phi(:,1:D1)+a_nu(:,k)*phi(k,1:D1)) * Phi(1,1,k);
        Phi(D1+1:end,D1+1:end,k) = (1/sigma_A2^2 + 1/sigma_n2^2 * sum(a_nu(:,k)))^-1 * eye(D2);
        phi(k,D1+1:end) = 1/sigma_n2^2 * a_nu(:,k)'*(a_X(:,D1+1:end)-a_nu*phi(:,D1+1:end)+a_nu(:,k)*phi(k,D1+1:end)) * Phi(D1+1,D1+1,k);
    end    
    
    % As per Algorithm 1 line 7,8, variational model parameters \nu & \tau are updated for each video individually    
    for i_img=1:xData.imgLength        
        
        cur_nu=cell2mat(c_nu(i_img,:)');
        cur_tau=cell2mat(c_tau(i_img,:)');
        
        if I<8 % change feature order here
            cur_nu = cur_nu(:,feature_order);
            cur_tau = [cur_tau(feature_order,:);cur_tau(K+1:end,:)];
        end
        
%         Reading features of all the tracks of curent video                
        cur_X=cell2mat(xData.xImgCellFlat(i_img,:)');

%         Reading weak labels of current video        
        cur_label=sum(cell2mat(xData.xTopicList(i_img,:)'),1);
        lengthOfLabel=length(cur_label);

        % latent index for backgrounds is set to 1. BG is present in all videos and tracks.        
        ws=[cur_label ones(1,K-lengthOfLabel)]; % lengthOfLabel
        ws = ws(feature_idx);

%         Reading information on correlated entities for using constraints        
        pairlist=xData.xPairList{i_img};
        
        % Using multinomial approximation (q) for optimisation on E_nu[log(1-prod(v_m))]
        q = zeros(K,K);
        q(1,1) = 1;
        for k = 2:K
            q(k,1) = exp(psi(cur_tau(1,2))-psi(cur_tau(1,1)+cur_tau(1,2)));
            for i = 2:k
                q(k,i) = exp(psi(cur_tau(i,2))+sum(psi(cur_tau(1:i-1,1)))-sum(psi(cur_tau(1:i,1)+cur_tau(1:i,2))));
            end
        end
        q = (q+eps)./repmat(sum(q,2),1,K);

        % Update \tau using equation S22 and S23 in supplementary
        for k = 1:K
            nu_sum = sum(cur_nu,1);
            cur_tau(k,1) = alpha + sum(nu_sum(k:K)) + (xData.segLenPerImg(i_img)-nu_sum(k+1:K))*sum(q(k+1:K,k+1:K),2);
            cur_tau(k,2) = 1 + (xData.segLenPerImg(i_img)-nu_sum(k:K))*q(k:K,k);
        end
        
        % Using multinomial approximation (q) for optimisation on E_nu[log(1-prod(v_m))]
        % Recomputing q based on updated tau
        q = zeros(K,K);
        q(1,1) = 1;
        for k = 2:K
            q(k,1) = exp(psi(cur_tau(1,2))-psi(cur_tau(1,1)+cur_tau(1,2)));
            for i = 2:k
                q(k,i) = exp(psi(cur_tau(i,2))+sum(psi(cur_tau(1:i-1,1)))-sum(psi(cur_tau(1:i,1)+cur_tau(1:i,2))));
            end
        end
        q = (q+eps)./repmat(sum(q,2),1,K);
        
        % Update nu
        % adding dummy column for simpler calculation
        cur_nu = [ones(size(cur_nu,1),1) cur_nu];
        rev_ordering = [0 rev_ordering];
        
        for k = 1:K
            tmpS = 0;
            if k > 1
                tmpS = fliplr(cumsum(fliplr(q(k,2:k))))*psi(cur_tau(1:k-1,1));
            end
            
            tmpC = 0;
            if(~isempty(pairlist(:,1) == feature_idx(k)+1))
                cnstrnt_numat = cur_nu(:,rev_ordering(pairlist((pairlist(:,1) == feature_idx(k)+1),2))+1);
                indx = find(cur_nu(:,k+1)'*cnstrnt_numat < 3);
                if(~isempty(indx))
                    tmpC = C*sum(cnstrnt_numat(:,indx),2);
                end
            end

            % Using Eq (11) and (12) to update \nu
            cur_nu(:,k+1) = (ws(k)>0)./(1+exp(-(...
                sum(psi(cur_tau(1:k,1))-psi(cur_tau(1:k,1)+cur_tau(1:k,2))) ...
                -(q(k,1:k)*psi(cur_tau(1:k,2)) + tmpS - fliplr(cumsum(fliplr(q(k,1:k))))*psi(cur_tau(1:k,1)+cur_tau(1:k,2)) - q(k,1:k)*log(q(k,1:k))')...
                - 0.5/sigma_n1^2*(trace(Phi(1:D1,1:D1,k))+phi(k,1:D1)*phi(k,1:D1)')...
                - 0.5/sigma_n2^2*(trace(Phi(D1+1:end,D1+1:end,k))+phi(k,D1+1:end)*phi(k,D1+1:end)')...
                + 1/sigma_n1^2*phi(k,1:D1)*(cur_X(:,1:D1)-cur_nu(:,2:end)*phi(:,1:D1)+cur_nu(:,k+1)*phi(k,1:D1))'...
                + 1/sigma_n2^2*phi(k,D1+1:end)*(cur_X(:,D1+1:end)-cur_nu(:,2:end)*phi(:,D1+1:end)+cur_nu(:,k+1)*phi(k,D1+1:end))' + tmpC')));
            
        end
        % removing dummy class         
        cur_nu(:,1)=[];
        rev_ordering(1) = [];        
        
%         Storing updated parameters back into input data handler
        c_tau(i_img,1:xData.segLenPerImg(i_img))= mat2cell(cur_tau,[K*ones(1,xData.segLenPerImg(i_img))],2)';
        c_nu(i_img,1:xData.segLenPerImg(i_img))= mat2cell(cur_nu,[ones(1,xData.segLenPerImg(i_img))],K)';

%         For the final iteration we undo all the mapping and store back.
%         This step maps back value of \nu to corresponding latent factors         
        if(I == para.MAX_ITERATIONS)
            c_nu(i_img,1:xData.segLenPerImg(i_img))= mat2cell(cur_nu(:,rev_ordering),[ones(1,xData.segLenPerImg(i_img))],K)';
            c_tau(i_img,1:xData.segLenPerImg(i_img))= mat2cell([cur_tau(rev_ordering,:);cur_tau(K+1:end,:)],[K*ones(1,xData.segLenPerImg(i_img))],2)';
        end
        
    end        

%   Track accuracy every iteration    
    if(I == para.MAX_ITERATIONS)
        [p_acc(I),a_acc(I),c_acc(I)] = accuracy(c_nu,1:K,xData,1);
%         Uncomment for a2d dataset
%         [p_acc(I),a_acc(I),c_acc(I),~,~] = accuracy_a2d(c_nu,1:K,xData,1,D1);
    else
        [p_acc(I),a_acc(I),c_acc(I)] = accuracy(c_nu,rev_ordering,xData,0);
%         Uncomment for a2d dataset   
%         [p_acc(I),a_acc(I),c_acc(I),~,~] = accuracy_a2d(c_nu,rev_ordering,xData,0,D1);
    end   
    fprintf('%d\t%1.4f\t%1.4f\t%1.4f\n',I,p_acc(I),a_acc(I),c_acc(I));
end

% Return back variational parameters \nu, \Phi and \sigma_k
% These are used during inference.
learnedModel.c_nu=c_nu;
learnedModel.phi=phi(rev_ordering,:);
learnedModel.Phi=Phi(:,:,rev_ordering);
end
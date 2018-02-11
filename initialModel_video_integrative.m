function [para,model] = initialModel_video_integrative(xData,D1)
para.D=size(xData.allX,2);
para.MAX_ITERATIONS=50;

para.D1=D1;
para.D2=para.D-D1;

% For appropriate parameters used, please refer paper.
% Cross validation between K = numclasses : 10 : numclasses + 100
para.K=30; % Set to some high number K >> numclasses

% Cross validation between C = 0 : 0.5 : 5
para.C=0.5;

para.sigmaA1=2;
para.sigmaN1=1;
para.sigmaA2=1;
para.sigmaN2=1;
para.alpha=100;

model.Phi=ones(para.D,para.D,para.K);
model.phi=zeros(para.K,para.D)/(para.K*para.D);

for iImg=1:xData.imgLength
    for jSeg=1:xData.segLenPerImg(iImg)
        % One can also randomly initialize \nu. We found no major diference in the accuracy using non-random initialization.
%         tem_nu=0.5+0.001*rand(1,para.K);
        tem_nu=0.5*ones(1,para.K);%/para.K;
        model.c_nu{iImg,jSeg}=tem_nu;
        model.c_tau{iImg,jSeg}=[ones(para.K,1)*para.alpha ones(para.K,1)/para.K];
    end
end

end


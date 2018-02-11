clear 
close all

% create dataset. The function for creating casablanca dataset is provided
length_video_clip = 60;
num_persons = 19;
dim1 = 32; % Dimension of subject data.
create_casablanca_dataset(length_video_clip,num_persons,dim1);

% Load dataset. Output needs to be structure in a variable xData. Please
% refer create_casablanca_dataset() function
filename = sprintf('data_%s_%s.mat',num2str(length_video_clip),num2str(num_persons));
load(filename); 

% For non-integrative models:-
% [para,model] = initialModel_video(xData);
% [learnedModel,p_acc(i,:),a_acc(i,:)] = wscsibp(para,model,xData);

% For Integrative models:-
[para,model] = initialModel_video_integrative(xData,dim1);

% [learnedModel,p_acc,a_acc,c_acc] = wssiibp(para,model,xData);
[learnedModel,p_acc,a_acc,c_acc] = wscsiibp(para,model,xData);


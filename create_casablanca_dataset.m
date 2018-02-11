function create_casablanca_dataset(length_video_clip,num_persons,dim)

% Enter the length of video clip in terms of number of seconds
load('Casablanca/data.mat');

for i = 1:max(GTf)
    f_person(i) = length(find(GTf==i));
end
[sorted_person personid] = sort(f_person,'descend');

% Retain the ground truths of only the <num_persons> most frequently occuring actors in the video, and set the ground truths of other actors equal to 1 (background class).
for i = 1:length(GTf)
    if(isempty(find(personid(2:num_persons+1)==GTf(i))))
        GTf(i) = 1;
    end
end

temp = zeros(size(GTf));
for i = 1:num_persons+1
    temp(GTf == personid(i)) = i;
end

% frame rate = 24 frames per second
num_videos = ceil(max(tframes(:))/(length_video_clip*24));

num_samples = 0;
for i = 1:num_videos
    for j = 1:size(tframes,1)
        track_video_overlap_value(j,i) = track_video_overlap(tframes(j,:),i, length_video_clip);
    end
    ind_i = find(track_video_overlap_value(:,i)==1);
    if(isempty(ind_i))
        continue;
    end
    num_samples = num_samples + 1;
    segLenPerImg(num_samples,1) = length(ind_i);
    for j=1:segLenPerImg(num_samples,1)
        s = zeros(1,(num_persons+2)+2);        
        s(temp(ind_i(j))) = 1;
        s(1) = [];
        s(GTa(ind_i(j))+num_persons) = 1;          
        s(num_persons+1) = [];
        x_topic_list{num_samples,j}=s;
        x_img_cell_flat{num_samples,j}=[Kf(ind_i(j),1:dim) Ka(ind_i(j),1:16)];
    end    
    p = unique((temp(ind_i)-1)*3 + GTa(ind_i));
    p = [ceil(p/3) mod(p-1,3)+num_persons+1];
    p(p(:,2) == num_persons+1,2) = 1;
    xPairList{num_samples} = [p;p(:,[2 1])];
end

xData.xImgCellFlat=x_img_cell_flat;
xData.xTopicList=x_topic_list;
xData.xPairList=xPairList;
xData.imgLength=num_samples;
xData.segLenPerImg=segLenPerImg;
xData.allSegLength=sum(segLenPerImg);
tempx=xData.xImgCellFlat';
xData.allX=cell2mat(tempx(:));
xData.num_persons = num_persons;
xData.num_actions = 2;

filename = sprintf('data_%s_%s.mat',num2str(length_video_clip),num2str(num_persons));
save(filename,'xData');
end
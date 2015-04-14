%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The dimensionality reduction and recap
%  
%
%  written by Tianpei Xie, Oct_30_2012
%  modified Nov_5_2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

curpath = pwd; 
[dest_org, foldername, ext] = fileparts(curpath);
%upper_org = '../../../Raw_data/Segmented_data';
upper_org2 = '../../Raw_data/Features_data'; %'../../../Raw_data/Features_data';
upper_org = upper_org2;

choice = 1;
ifsimu = 1;

if choice == 1
src_org =  strcat(upper_org, '/MFCC_win32ms');
dest_org = strcat(upper_org2, '/MFCC_win32ms_dim_red');
elseif choice == 1 && ifsimu == 1
src_org =  strcat(upper_org, '/MFCC_win32ms');
dest_org = strcat(upper_org2, '/MFCC_win32ms_dim_sim');
elseif choice ==3
src_org =  strcat(upper_org, '/Spec_win32ms');
dest_org = strcat(upper_org2, '/Spec_win32ms_dim_red');  
elseif choice == 4
src_org =  strcat(upper_org, '/MFCC_win32ms_envelop');
dest_org = strcat(upper_org2, '/MFCC_win32ms_envelop_dim_red'); 
else 
src_org =  strcat(upper_org, '/PLP_envelop');
dest_org = strcat(upper_org2, '/PLP_envelop_dim_red');     
end
src_dir = [{'/human_09/'}, ...
       {'/human_animal_09/'}, ...
       {'/human_10/'},...
       {'/human_animal_10/'} ];
   
dest_dir =  '/Dictionary/';
 
      
testChan  = [1:8, 10]; %[1 2 3 4 ];
fs = 10000/10;

red_dim = 100;%20; %reduced dimensionality
dataset_temp = cell(1,length(src_dir));
dataset_info_temp = cell(1,length(src_dir));
class_label_temp = cell(1,length(src_dir));
%%
for folderid = 1:length(src_dir)
% loop over folders
    pathOrig = strcat(src_org, src_dir{folderid});
    pathDest = strcat(dest_org, dest_dir);
    listAllFiles = dir(pathOrig);
    nFiles = length(listAllFiles);
    display(['=====================================================']);
    display(['In folder ' pathDest]);
    display(['In folder ' pathOrig]);
    display(['Writing the log file...']);
% Create/Write the log file
    fid = fopen(strcat(pathDest ,'/log.txt'), 'w+');
    fprintf(fid, ' \n');
    fprintf(fid, '==================================================\n');
    time = clock;
    fprintf(fid, 'Date:\t %s\t Time:\t %d:%d:%2d \n', date, time(4),time(5),ceil(time(6)));
    fprintf(fid, 'CurrentPath:\t %s\n', curpath);
    fprintf(fid, 'Src_Filepath:\t  %s\n', pathOrig);
    fprintf(fid, 'Dest_Filepath:\t  %s\n', pathDest);
    fprintf(fid, 'Activity:\t  dimensionality reduction: MFCC with filtering  r = %d \n',red_dim);
    %fprintf(fid, 'Activity:\t  dimensionality reduction: MFCC with filtering  \n');
    fprintf(fid, '-------------------------------------------------------------\n');
    %fprintf(fid, 'Parameter:\n');
    %fprintf(fid, '\t Window time(MFCC) (.sec): \t %.3f \t Hoptime: \t %.3f \n', wintime, hoptime);
    %fprintf(fid, '\t Num of cepstal returned: \t %d \n', numcep);
    %fprintf(fid, '\t Num of filter band: \t %d \n', nbands);
    %fprintf(fid, '\t Maximal freq of filter band (Hz): \t %d \n', maxfreq);
    fprintf(fid, '-------------------------------------------------------------\n');
    %fprintf(fid, 'Src_fullname \t  Dst_fullname \t Dimension of feature \t norm of feature \n ');
    %fprintf(fid, '-------------------------------------------------------------\n');

    ncol = nFiles-3;
    dataset_temp{folderid} = cell(length(testChan),ceil(ncol/length(testChan)));
    dataset_info_temp{folderid} = repmat( struct('index', 1, 'preName', [],...
                             'Chan', 1, 'Seg', 1, 'Part', 1, ...
                          'ishuman', 1, 'iscorrupted', 0,'day',1209), ...
                          length(testChan),ceil(ncol/length(testChan)));
                      
    class_label_temp{folderid} = ones(length(testChan), ceil(ncol/length(testChan)));
    fprintf(fid, 'dataset name \t  num of instances\n ');
    fprintf(fid, '%s \t %d\n',src_dir{folderid}, ncol);
    count = ones(length(testChan),1);
    prechanNum = 10;
  for i = 1:nFiles
    fullname = listAllFiles(i).name;        
    
    
    
    if strfind(fullname,'mat') ~= 0
        filename = fullname(1:end-4);
        
        
        
        indx = strfind(filename,'_');
        % Get the date
        preName = filename(1:indx(2)-1);
        day = str2num(filename(7:8));
        % Get channel number
        chanNum = str2num(filename(indx(2)+5:indx(3)-1));
        % Get Segment number
        indx2 = strfind(filename,'Fea');
        segNum = str2num(filename(indx2+3:end));
        % Get Part number
        indx3 = strfind(filename,'part');
        partNum = str2num(filename(indx3+4));
        
        if folderid==1 || folderid==3
            ishuman = 1;
        else
            ishuman = -1;
        end
        [iscorrupt, index ]= corrupt_infer(preName, chanNum);
        
        info_struct = struct('index', 2*(index-1)+partNum, 'preName', preName,...
                             'Chan', chanNum, 'Seg', segNum, 'Part', partNum, ...
                          'ishuman', ishuman, 'iscorrupted', iscorrupt, 'day',day);
        
        
        
        fullpath = strcat(pathOrig, fullname)
            
            
        load(fullpath)
        ind_row = find(ismember(testChan,chanNum));
%         if count(ind_row)==937 
%             pause
%         end
        dataset_temp{folderid}{ind_row, count(ind_row)} = fea;
        dataset_info_temp{folderid}(ind_row, count(ind_row)) = info_struct;
        class_label_temp{folderid}(ind_row, count(ind_row)) = ishuman;
        count(ind_row) = count(ind_row)+1;
            
           display('end of feature collection ');
           
           clear data  
      %fprintf(fid, '---------------------------------------------------------\n');
    end
  end
 
end


%% Data collection
dataset = [];
dataset_info = [];
class_label = [];

display('Gathering data instances...');
for folderid =1:length(src_dir)
dataset = [dataset dataset_temp{folderid}];
dataset_info = [dataset_info, dataset_info_temp{folderid}];
class_label = [class_label, class_label_temp{folderid}];
end

% len_r = length(dataset{1,1});
% 
% ind_t = [];
% for i=1:length(dataset(1,:))
% fea = dataset{1,i}; 
%  if length(fea) ~= len_r 
%   ind_t = [ind_t, i];
%  end
% end
% 
% dataset(:,ind_t) = [];
% dataset_info(:,ind_t) = [];
% class_label(:, ind_t) = [];


ind_c1 = find(class_label(1,:) == 1);
ind_c2 = find(class_label(1,:) == -1); 

dataset_org = cell(length(testChan),1);
mean_class = cell(length(testChan),2);
for ii = 1:length(testChan)
dataset_org{ii} = cell2mat(dataset(ii,:));
 % mean vector for the data
 mean_class{ii,1} = mean(dataset_org{ii}(:,ind_c1),2);  
 mean_class{ii,2} = mean(dataset_org{ii}(:,ind_c2),2);
end

fprintf(fid, 'original dim \t  reduced dim \t total instances\n ');
fprintf(fid, '%d \t  %d \t %d\n ', size(dataset_org{1},1), ...
                red_dim, size(dataset_org{1},2)*length(testChan));
datasetname = strcat(pathDest,'dataset_org.mat');
save(datasetname, 'dataset_org', 'mean_class'); 


%% Standarization
display('standarization...');
org_dim = size(dataset_org,1);
dataset_new =  cell(length(testChan),1);
for ii = 1:length(testChan)
 dataset_new{ii} = dataset_org{ii};%zscore(dataset_org{ii}')'; 
end

for ii = 1:length(testChan)
 dataset_new{ii} = zscore(dataset_org{ii}')'; 
end

% for ii = 1:length(testChan)
%   for p =1:org_dim
%    %dataset_new{ii}(p,ind_c1) = dataset_new{ii}(p,ind_c1) - mean(dataset_new{ii}(p,ind_c1));
%    %dataset_new{ii}(p,ind_c2) = dataset_new{ii}(p,ind_c2) - mean(dataset_new{ii}(p,ind_c2));
%    %dataset_new(p,:) = dataset_new(p,:)./sqrt(var(dataset_new(p,:)));
%    dataset_new{ii}(p,:) = dataset_new{ii}(p,:) - mean(dataset_new{ii}(p,:));
%   end
% end
%%
iscorrupt = zeros(size(dataset_info));
index     = zeros(size(dataset_info));
name      = zeros(size(dataset_info));
dayarray       = zeros(size(dataset_info));

for i=1:length(testChan)
   for j=1:size(dataset_info,2)
       iscorrupt(i,j) = dataset_info(i,j).iscorrupted;
       index(i,j) = dataset_info(i,j).index;
       name_temp =   dataset_info(i,j).preName;
       name(i,j) = str2num(name_temp(7:8));
       dayarray(i,j) = dataset_info(i,j).day;
   end
end

ind_c1_09 = intersect(find(dayarray(1,:)==09),ind_c1);
ind_c2_09 = intersect(find(dayarray(1,:)==09),ind_c2);
ind_c1_10 = intersect(find(dayarray(1,:)==10),ind_c1);
ind_c2_10 = intersect(find(dayarray(1,:)==10),ind_c2);


ind_c1_dirty = cell(length(testChan),1);
ind_c2_dirty = cell(length(testChan),1);
ind_c1_clean = cell(length(testChan),1);
ind_c2_clean = cell(length(testChan),1);
for i=1:length(testChan)
ind_c1_clean{i} = intersect(ind_c1,find(iscorrupt(i,:)==0));
ind_c2_clean{i} = intersect(ind_c2,find(iscorrupt(i,:)==0));
ind_c1_dirty{i} = intersect(ind_c1,find(iscorrupt(i,:)==1));
ind_c2_dirty{i} = intersect(ind_c2,find(iscorrupt(i,:)==1));
end
%%
datasetname = strcat(pathDest,'dataset_new.mat');
save(datasetname, 'dataset_new', 'dataset_info', 'class_label'); 

display('Dimensionality reduction...');
%pca_mat = dataset_new*dataset_new'./size(dataset_new,2);
%[pcacoeff,latent,explained] = pcacov(pca_mat);

pcacoeff = cell(length(testChan),1);
latent= cell(length(testChan),1);
tsquare = cell(length(testChan),1);

pca_proj = pcacoeff;
dataset_new_red = cell(length(testChan),1);
ratio = cell(length(testChan),1);
cutoff_dim = zeros(length(testChan),1);

for ii = 1:length(testChan)
[pcacoeff{ii}, ~, latent{ii}, tsquare{ii}]= princomp(dataset_new{ii}');
 
 %h1= figure(10);
 %plot(1:length(latent{ii}))
   count2 =0;
   norm2 = sum(latent{ii});
   ratio{ii} = zeros(1,length(latent{ii}));
   for k=1:length(latent{ii})
       count2 = count2 + latent{ii}(k);
       ratio{ii}(k) = count2/norm2;
       if ratio{ii}(k) > 0.8
           break;
       end
   end
   cutoff_dim(ii) = k;
 
pca_proj{ii} = pcacoeff{ii}(:,1:red_dim);
dataset_new_red{ii} = pca_proj{ii}'*dataset_new{ii};

diff_mean = mean_class{ii,1} - mean_class{ii,2};
alpha =0.2;
% run for simulation
if ifsimu == 1
    if ismember(ii, [1:2,5:9])
    dataset_new_red{ii}(:,ind_c1) = dataset_new_red{ii}(:,ind_c1) ...
        + pca_proj{ii}'*repmat(mean_class{ii,1} - alpha*diff_mean ,1,length(ind_c1));
    dataset_new_red{ii}(:,ind_c2) = dataset_new_red{ii}(:,ind_c2) ...
        + pca_proj{ii}'*repmat(mean_class{ii,2}  ,1,length(ind_c2));
    else
    dataset_new_red{ii}(:,ind_c1_clean{ii}) = dataset_new_red{ii}(:,ind_c1_clean{ii}) ...
    + pca_proj{ii}'*repmat(mean_class{ii,1}- alpha*diff_mean ,1,length(ind_c1_clean{ii}));  
    dataset_new_red{ii}(:,ind_c1_dirty{ii}) = dataset_new_red{ii}(:,ind_c1_dirty{ii}) ...
    + pca_proj{ii}'*repmat(mean([mean_class{ii,1}, mean_class{ii,2}],2),1,length(ind_c1_dirty{ii})); 
    dataset_new_red{ii}(:,ind_c2_clean{ii}) = dataset_new_red{ii}(:,ind_c2_clean{ii}) ...
    + pca_proj{ii}'*repmat(mean_class{ii,2},1,length(ind_c2_clean{ii}));
    dataset_new_red{ii}(:,ind_c2_dirty{ii}) = dataset_new_red{ii}(:,ind_c2_dirty{ii}) ...
    + pca_proj{ii}'*repmat(mean([mean_class{ii,1}, mean_class{ii,2}],2),1,length(ind_c2_dirty{ii})); 
    end
end
end

datasetname = strcat(pathDest,'dataset_new_red.mat');
hist  = cell(length(testChan),1);
hist_clean = hist;

W_lda = cell(length(testChan),1);
dataset_lda = cell(length(testChan),1);
bin = cell(length(testChan),1);

S1 =zeros(size(dataset_org{ii},1));
S2 =zeros(size(dataset_org{ii},1));

%% Fisher LDA score 
  for ii = 1:length(testChan)
  S1 = cov(dataset_org{ii}(:,ind_c1)');
  S2 = cov(dataset_org{ii}(:,ind_c2)');
  SW = S1 + S2;
  v = mean(dataset_org{ii}(:,ind_c2),2) - mean(dataset_org{ii}(:,ind_c1),2);
 % mean vector for the data
  W_lda{ii} = SW\v; 
  end
  
  
  for ii = 1:length(testChan)
  dataset_lda{ii} = W_lda{ii}'*dataset_org{ii};
  bin{ii} = [min(dataset_lda{ii}):0.2:max(dataset_lda{ii})];
  
  k11 = kurtosis(dataset_lda{ii}(ind_c1))
  k12 = kurtosis(dataset_lda{ii}(ind_c2)) 
  
  hist{ii}(1,:) = histc(dataset_lda{ii}(ind_c1),bin{ii});
  hist{ii}(2,:) = histc(dataset_lda{ii}(ind_c2),bin{ii});
  
  k1 = kurtosis(dataset_lda{ii}(ind_c1_clean{ii}))
  k2 = kurtosis(dataset_lda{ii}(ind_c2_clean{ii})) 
  
  hist_clean{ii}(1,:) = histc(dataset_lda{ii}(ind_c1_clean{ii}),bin{ii});
  hist_clean{ii}(2,:) = histc(dataset_lda{ii}(ind_c2_clean{ii}),bin{ii});
  end

%% store data
save(datasetname, 'dataset_new_red', 'pcacoeff', 'latent', 'tsquare',...
    'dataset_lda','W_lda', 'hist','hist_clean','bin', 'cutoff_dim', 'ratio'); 
fprintf(fid, '----------------------end of log--------------------\n ');
  fprintf(fid, ' \n');
  fclose(fid);

%% plot 
   index2 = zeros(size(dataset_info));
   for i=1:size(dataset_info,1)
      for j=1:size(dataset_info,2)
        index2(i,j) = dataset_info(i,j).index;
      end 
   end
   
   ii = 3;
   figure(2);
   show_dataset = dataset_new_red{ii};
   plot(show_dataset(1,:), show_dataset(2,:), 'or');
   grid on;
   xlabel('1st pca');
   ylabel('2nd pca');
% %axis([-10, 10, -10, 10]);
  title('reduced feature')
   
   
  
  
    figure(3);
   show_dataset = dataset_new_red{ii};
%    proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
%    proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
%    show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
%    show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
   plot(show_dataset(1,ind_c1), show_dataset(2,ind_c1), 'or', 'Linewidth',1.5);
   grid on;
   hold on;
   plot(show_dataset(1,ind_c2), show_dataset(2,ind_c2), 'xb');
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human', 'human-animal')

  figure(3);
   show_dataset = dataset_new_red{ii};
   %proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
   %proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
   %show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
   %show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
   plot(show_dataset(1,ind_c1_09), show_dataset(2,ind_c1_09), 'or', 'Linewidth',1.5);
   grid on;
   hold on;
   plot(show_dataset(1,ind_c2_09), show_dataset(2,ind_c2_09), 'xb','Linewidth',1.5);
   plot(show_dataset(1,ind_c1_10), show_dataset(2,ind_c1_10), 'om', 'Linewidth',1);
    plot(show_dataset(1,ind_c2_10), show_dataset(2,ind_c2_10), 'xg', 'Linewidth',1);
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('clean reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human 09', 'human-animal 09','human 10','human-animal 10')

    figure(4);
   show_dataset = dataset_new_red{ii};
   %proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
   %proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
   %show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
   %show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
   plot(show_dataset(1,ind_c1_clean{ii}), show_dataset(2,ind_c1_clean{ii}), 'or', 'Linewidth',1);
   grid on;
   hold on;
   plot(show_dataset(1,ind_c2_clean{ii}), show_dataset(2,ind_c2_clean{ii}), 'xb');
   plot(show_dataset(1,ind_c1_dirty{ii}), show_dataset(2,ind_c1_dirty{ii}), 'om', 'Linewidth',1);
    plot(show_dataset(1,ind_c2_dirty{ii}), show_dataset(2,ind_c2_dirty{ii}), 'xg', 'Linewidth',1);
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('clean reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human', 'human-animal','human-dirty','human-animal-dirty')
  
  
  figure(9)
   show_dataset = dataset_new_red{ii};
   %proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
   %proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
   %show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
   %show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
   plot(show_dataset(1,ind_c1_09), show_dataset(2,ind_c1_09), 'or', 'Linewidth',1);
   grid on;
   hold on;
   plot(show_dataset(1,ind_c2_09), show_dataset(2,ind_c2_09), 'xb','Linewidth',1);
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('clean reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human 09', 'human-animal 09')
  
  figure(10)
   show_dataset = dataset_new_red{ii};
   %proj_mean_1  = pca_proj{ii}'*mean_class{ii,1};
   %proj_mean_2  = pca_proj{ii}'*mean_class{ii,2};
   %show_dataset(:,ind_c1)= show_dataset(:,ind_c1) + repmat(proj_mean_1,1,length(ind_c1));
   %show_dataset(:,ind_c2)= show_dataset(:,ind_c2) + repmat(proj_mean_2,1,length(ind_c2));
   
      plot(show_dataset(1,ind_c1_10), show_dataset(2,ind_c1_10), 'om', 'Linewidth',1);
   grid on;
   hold on;
    plot(show_dataset(1,ind_c2_10), show_dataset(2,ind_c2_10), 'xg', 'Linewidth',1);
   hold off;
   xlabel('1st pca');
   ylabel('2nd pca'); 
   if ismember(choice, [1,2,4]) 
      type = 'mfcc';
     
   else
       type = 'plp';
   end
    title(sprintf('clean reduced feature (%s) sensor %d', type, testChan(ii)));
  legend('human 10','human-animal 10')
  
  %%
  
  figure(5)
  bar(hist{ii}','histc');
  xlabel 'bin index';
  ylabel 'histogram';
  title(sprintf('histogram of Fisher LDA score for sensor %d',testChan(ii)));
  legend('human','human-animal')
  
  figure(6)
  bar(hist_clean{ii}','histc');
  xlabel 'bin index';
  ylabel 'histogram';
  title(sprintf('clean histogram of Fisher LDA score for clean sensor %d',testChan(ii)));
  legend('human','human-animal')
  
  
  %%
  
S11 =zeros(size(dataset_new_red{ii},1));
S12 =zeros(size(dataset_new_red{ii},1));
hist2  = cell(length(testChan),1);
hist_clean2 = hist2;

W_lda2 = cell(length(testChan),1);
dataset_lda2 = cell(length(testChan),1);
bin2 = cell(length(testChan),1);

%% Fisher LDA score 
  for ii = 1:length(testChan)
  S11 = cov(dataset_new_red{ii}(:,ind_c1)');
  S12 = cov(dataset_new_red{ii}(:,ind_c2)');
  SW2 = S11 + S12;
  v2 = mean(dataset_new_red{ii}(:,ind_c2),2) - mean(dataset_new_red{ii}(:,ind_c1),2);
 % mean vector for the data
  W_lda2{ii} = SW2\v2; 
  end
  
  
  for ii = 1:length(testChan)
  dataset_lda2{ii} = W_lda2{ii}'*dataset_new_red{ii};
  bin2{ii} = [min(dataset_lda2{ii}):0.2:max(dataset_lda2{ii})];
  
  k11 = kurtosis(dataset_lda2{ii}(ind_c1))
  k12 = kurtosis(dataset_lda2{ii}(ind_c2)) 
  
  hist2{ii}(1,:) = histc(dataset_lda2{ii}(ind_c1),bin2{ii});
  hist2{ii}(2,:) = histc(dataset_lda2{ii}(ind_c2),bin2{ii});
  
  k1 = kurtosis(dataset_lda{ii}(ind_c1_clean{ii}))
  k2 = kurtosis(dataset_lda{ii}(ind_c2_clean{ii})) 
  
  hist_clean2{ii}(1,:) = histc(dataset_lda2{ii}(ind_c1_clean{ii}),bin2{ii});
  hist_clean2{ii}(2,:) = histc(dataset_lda2{ii}(ind_c2_clean{ii}),bin2{ii});
  end
 %% 
  ii = 1
  figure(7)
  bar(hist2{ii}','histc');
  xlabel 'bin index';
  ylabel 'histogram';
  title(sprintf('histogram of Fisher LDA score for sensor %d',testChan(ii)));
  legend('human','human-animal')
  
  figure(8)
  bar(hist_clean2{ii}','histc');
  xlabel 'bin index';
  ylabel 'histogram';
  title(sprintf('clean histogram of Fisher LDA score for clean sensor %d',testChan(ii)));
  legend('human','human-animal')
  
  
  
  feas = [];
  id= [10 15];
  for k=1:length(id)
   for ii=1:4
     feas = [feas dataset_new_red{ii}(:,id(k))];  
   end
  end
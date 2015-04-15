%% Test on Internet Advertisement/WebKB dataset

clearvars
close all
clc

addpath('./libsvm-3.18/matlab/')
addpath('../../../../../../MATLAB/cvx/');

%load('InternetAds_data.mat');
load('WebKB_data.mat')

nV = 2;

TotalRun = 1; %20;

accuracy_v1_array = zeros( 20, length(nL_ex));
accuracy_v2_array = zeros( 20, length(nL_ex));
accuracy_mv_array = zeros( 20, length(nL_ex));
accuracy_ct_array = zeros( 20, length(nL_ex));
accuracy_svm2k_array= zeros( 20, length(nL_ex));
accuracy_mvmedSun_array= zeros( 20, length(nL_ex));


for idL = 1%1:length(nL_ex)
idxLabsTrn = indLabs_ex(:,idL);
nL = size(idxLabsTrn{1},2);
nU = size(DataX{1},1) - nL;
nTst = nU; % use the unlabeled set as test set

 display(sprintf('\n==========================================================='))
display(sprintf('Training size(labeled) %d:', nL));
display(sprintf('Training size(unlabeled) %d:', nU));
display(sprintf('Testing size %d:', nTst));
%% dataset and parameter setting
sigma = [1,1];%[2, 2];
% for initial SVM
sel =   2; % use Gaussian Kernel  %0; % use linear
% for mv-med
    param.regParam = 1; 
    for iv=1:nV
      param.kernelParm(iv)   =  sigma(iv); %sqrt(1./(2*sigma(iv)^2));
      param.kernelMethod{iv} =  {'rbf'}; %{'linear'};   %
    end
    
   % if idL <3
       param.maxIterOut = 1; %20; %1; %20;
   % else
    %    param.maxIterOut = 1;
   % end
    param.threOut = 1e-3;
    param.maxIterMAP = 50;
    param.threMAP = 1e-4;
    if idL ==1
        param.sigmaPri = 1.5;% 0.5; %1.5;
    else
        param.sigmaPri = 1;
    end
    param.mode = 1;
    param.q_thresh = 0.55; %0.6;

  % for co-training
   param_ct.nsel = 3;
   param_ct.maxIterOut = 200;
   param_ct.psel = 3;
   param_ct.rpool = 0.5;
   param_ct.mode = 1;
   param_ct.Distribution = [{'mn'}, {'mn'}]; % multinomial for Bayes classifier
   


%%
% cross validation
for R= 1%1:20 %size(idxLabs,1)
    display(sprintf('Repeat experiments %d:', R));

    optionsvm = cell(1,nV); 
    gamma_org = zeros(1,nV);
    
    % check uniqueness of index
    flagadd = 0;
    for iv=1:nV
     [C, indC] = unique(idxLabsTrn{iv}(R,:));
     if length(C)~= length(idxLabsTrn{iv}(R,:)) && flagadd == 0
         display('duplicted index inclueded.');
         inddup = setdiff([1:length(idxLabsTrn{iv}(R,:))], indC);
         poolAdd = setdiff([1:size(DataX{iv},1)], C);
         display('resampling ... ');
         addmore = randsample(poolAdd, length(inddup));
         for ivv=1:nV
            idxLabsTrn{ivv}(R,inddup) = addmore; 
         end
         flagadd = 1;
     end
     C = [];
     indC = [];
    end
    
    Traindata.X_L{1} = DataX{1}(idxLabsTrn{1}(R,:),:);
    Traindata.X_L{2} = DataX{2}(idxLabsTrn{2}(R,:),:) ;
    Traindata.y_L = DataY(idxLabsTrn{1}(R,:)) ;
    
    Traindata.X_U{1} = DataX{1}( setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:)) ,:) ;
    Traindata.X_U{2} = DataX{2}( setdiff([1:size(DataX{2},1)],idxLabsTrn{2}(R,:)) ,:) ;
    Traindata.y_U = DataY(setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:))) ;
  
    Traindata.nU = nU;
    Traindata.nL = nL;
    Traindata.nV = nV;
    Traindata.d = [size(DataX{1},2),size(DataX{2},2)];

    Testdata.X_Tst{1} = DataX{1}( setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:)) ,:) ;
    Testdata.X_Tst{2} = DataX{2}( setdiff([1:size(DataX{2},1)],idxLabsTrn{2}(R,:)) ,:) ;
    Testdata.y_Tst = DataY(setdiff([1:size(DataX{1},1)],idxLabsTrn{1}(R,:))) ;
    Testdata.nTst  = nTst;
    Testdata.d = [size(DataX{1},2),size(DataX{2},2)];
    TempXv1 = [];
    TempXv2 = [];
    epsilon_l = 1e-3;
    
%% adding feature noise to fit Gaussian Kernel    
    Traindata_mod = Traindata;
    Testdata_mod = Testdata;
%    for iv=1:nV
% %     TempXv1 = [Traindata.X_L{iv}(find(Traindata.y_L == 1),:)];
% %     TempXv2 = [Traindata.X_L{iv}(find(Traindata.y_L == -1),:)];
% %     TempX = TempX(:,any(TempX));
% %     Traindata_mod.X_L{iv}=   normr(Traindata_mod.X_L{iv});%+ epsilon_l*ones(size(Traindata_mod.X_L{iv}));
% %     Traindata_mod.X_U{iv}=   normr(Traindata_mod.X_U{iv});% + epsilon_l*ones(size(Traindata_mod.X_U{iv}));
% %     Testdata_mod.X_Tst{iv} = normr(Testdata_mod.X_Tst{iv});%+ epsilon_l*ones(size(Testdata_mod.X_Tst{iv}));
%     Traindata_mod.X_L{iv}=   Traindata_mod.X_L{iv}*Traindata.d(iv);%+ epsilon_l*ones(size(Traindata_mod.X_L{iv}));
%     Traindata_mod.X_U{iv}=   Traindata_mod.X_U{iv}*Traindata.d(iv);% + epsilon_l*ones(size(Traindata_mod.X_U{iv}));
%     Testdata_mod.X_Tst{iv} = Testdata_mod.X_Tst{iv}*Traindata.d(iv);%+ epsilon_l*ones(size(Testdata_mod.X_Tst{iv}));
%  
% %     TempXv1 = [];
% %     TempXv2 = [];
%    end

for iv=1:nV
    if strcmp(param.kernelMethod{iv}, 'chisquare')
     K_l{iv} = pdist2(Traindata_mod.X_L{iv},Traindata_mod.X_L{iv}, @chi_square_statistics_fast); 
%     for n=1:nL
%        K_l{iv}(n,:)= histogram_intersection(Traindata_mod.X_L{iv}(n,:),Traindata_mod.X_L{iv});
%     end
    Traindata_mod.K{iv} = [ (1:nL)' , K_l{iv} ]; 
    K_lTst{iv} = pdist2(Testdata_mod.X_Tst{iv},Traindata_mod.X_L{iv}, @chi_square_statistics_fast); 
%      for n=1:nTst
%        K_lTst{iv}(n,:)= histogram_intersection(Testdata_mod.X_Tst{iv}(n,:),Traindata_mod.X_L{iv});
%      end
    Testdata_mod.K_lTst{iv} = [ (1:nTst)' , K_lTst{iv} ]; 
    end
end    
    
%%  Train initial model 

    for iv=1:nV
      gamma_org(iv) = 1./(2*sigma(iv)^2);
      optionsvm{iv} = sprintf('-t %d -c 1 -g %f', sel ,gamma_org(iv));
    end
    
    model = cell(1,nV);
    accuracy_v = zeros(3,nV);
    decision_valU = zeros(nU, nV);
    decision_valL = zeros(nL, nV);
    decision_valTst = zeros(nTst, nV);
    
    % train two view independent classifiers y_L
    for iv = 1:nV
        if strcmp(param.kernelMethod{iv}, 'hist')     
           model{iv} = svmtrain(Traindata_mod.y_L, Traindata_mod.K{iv}, optionsvm{iv}); 
           [~, accuracy_v(:,iv), ~] = svmpredict(Testdata_mod.y_Tst, Testdata_mod.K_lTst{iv}, model{iv});
        else
           model{iv} = svmtrain(Traindata_mod.y_L, Traindata_mod.X_L{iv},optionsvm{iv});
            [~, accuracy_vU(:,iv), decision_valL(:,iv)] = svmpredict(Traindata_mod.y_L,  Traindata_mod.X_L{iv}, model{iv});
            [~, accuracy_vU(:,iv), decision_valU(:,iv)] = svmpredict(Traindata_mod.y_U,  Traindata_mod.X_U{iv}, model{iv});
            [~, accuracy_v(:,iv), decision_valTst(:,iv)] = svmpredict(Testdata_mod.y_Tst, Testdata_mod.X_Tst{iv}, model{iv}); 
            model{iv}.decision_valL = decision_valL(:,iv);
            model{iv}.decision_valU = decision_valU(:,iv);
            model{iv}.decision_valTst = decision_valTst(:,iv);
        end
    end
  
    accuracy_v1_array(R,idL) = accuracy_v(1,1);
    accuracy_v2_array(R,idL) = accuracy_v(1,2);

%     %for iv=1:nV
%    options{1} = struct('Kernel', param.kernelMethod{1}, 'KernelParam', param.kernelParm(1)); 
% %end
%     
%      Kl_temp = calckernel(options{1},full(model{1}.SVs),X_L{1});
%      Ku_temp = calckernel(options{1},full(model{1}.SVs),X_U{1});
%     % SVM Prediction
%      Ktst_temp = calckernel(options{1},full(model{1}.SVs),X_Tst{i});
%        fl0(:1) = Kl_temp*model{1}.sv_coef- model{1}.rho; %; %Kl_temp*model{i}.sv_coef; Kl_temp*model{i}.sv_coef- model{i}.rho; %
%       fu0(:,1) =  Ku_temp*model{1}.sv_coef- model{1}.rho; %Ku_temp*model{i}.sv_coef;%model{iv}.decision_valU; 
%       ftst0(:,1) =  Ktst_temp*model{1}.sv_coef- model{1}.rho; %Ktst_temp*model{i}.sv_coef;%model{iv}.decision_valTst;
%     
%     for i=1:nV
%         p(:,i) = sigmoid(fu0(:,i));
%         if strcmp(options{i}.Kernel, 'linear')
%          p(:,i) = sigmoid(fu0(:,i)- model{i}.rho);  
%         end
%        errorlistp = (sign(p(:,i) - 0.5*ones(nTst,1))~= y_Tst) ; 
%        accuracyp(i) = 1 - sum(errorlistp)/nTst;
%        errorlistp= [];      
%         % CHANGE HERE
%         % p(:,i) = (sign(p(:,i))+1)/2;
%     end
      
    
    iniparam.inimodel = model;
    
%     options = cell(1,nV); 
%  for iv =1:nV 
%     
%    options{iv} = struct('Kernel', param.kernelMethod{iv}, 'KernelParam', param.kernelParm(iv)); 
%        Kl_temp = calckernel(options{iv},full(model{iv}.SVs),Traindata_mod.X_L{iv});
%       Ku_temp = calckernel(options{iv},full(model{iv}.SVs),Traindata_mod.X_U{iv});
%       Ktst_temp = calckernel(options{iv},full(model{iv}.SVs),Testdata_mod.X_Tst{iv});
%       fl0(:,iv) = Kl_temp*model{iv}.sv_coef;
%       fu0(:,iv) = Ku_temp*model{iv}.sv_coef;
%       ftst0(:,iv) = Ktst_temp*model{iv}.sv_coef;
%       
%       fl0(:,iv) = fl0(:,iv)- model{iv}.rho;  %add bias
%       fu0(:,iv) = fu0(:,iv)- model{iv}.rho;
%       ftst0(:,iv) = ftst0(:,iv) - model{iv}.rho;
%       
%       
%       errorlist = (sign(ftst0(:,iv))~= Testdata_mod.y_Tst) ; 
%   %errorlist = (sign(label_pred*view_weight)~= y_Tst) ; 
%   accuracy = 1 - sum(errorlist)/nTst
%  end   
    
%% Train with Mv-MED
% [accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_tr,...
%     history2, history_tst2] = mvmedbin(Traindata_mod, Testdata_mod, param, iniparam); 
%if idL == 1
%[accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_tr,...
%    history2, history_tst2] = mvmedbin_v1(Traindata_mod, Testdata_mod, param, iniparam); 
%else
% [accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_trn,...
%    history, history_tst] = mvmedbin_v3(Traindata_mod, Testdata_mod, param, iniparam); 
%end

% [accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_trn,...
%    history, history_tst] = mvmedbin_v4(Traindata_mod, Testdata_mod, param, iniparam); 

 [accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_trn,...
    history, history_tst] = mvmed_compInfo(Traindata_mod, Testdata_mod, param, iniparam); 

sigmoid = @(x)(1./(1+ exp(-x)));
p = sigmoid(dec_struct.mean);
 errorlist1 = (sign(p(:,1) - 0.5*ones(nTst,1))~= Testdata_mod.y_Tst) ; 
  %errorlist = (sign(label_pred*view_weight)~= y_Tst) ; 
  accuracy1 = 1 - sum(errorlist1)/nTst;
   errorlist2= (sign(p(:,2) - 0.5*ones(nTst,1))~= Testdata_mod.y_Tst) ; 
  %errorlist = (sign(label_pred*view_weight)~= y_Tst) ; 
  accuracy2 = 1 - sum(errorlist2)/nTst;

  display(sprintf('Best among three %.1f %%', max([accuracy, accuracy1, accuracy2])*100));
  
accuracy_mv_array(R,idL) = max([accuracy, accuracy1, accuracy2])*100; 
    

%% Train with co-training

% before submitting, the zero-frequency feature should be tackled. i.e. 
% Add 1 to the count for every attribute value-class combination 
% (Laplace estimator) when an attribute value (Outlook=Overcast) doesn�t occur with every class value
%Traindata_mod = Traindata;
%Testdata_mod = Testdata;

% for iv=1:nV
%     TempXv1 = [Traindata.X_L{iv}(find(Traindata.y_L == 1),:)];
%     TempXv2 = [Traindata.X_L{iv}(find(Traindata.y_L == -1),:)];
%     TempX = TempX(:,any(TempX));
%     Traindata_mod.X_L{iv}=   round(Traindata_mod.X_L{iv}./epsilon_l+ ones(size(Traindata_mod.X_L{iv})));
%     Traindata_mod.X_U{iv}=   round(Traindata_mod.X_U{iv}./epsilon_l +ones(size(Traindata_mod.X_U{iv})));
%     Testdata_mod.X_Tst{iv} = round(Testdata_mod.X_Tst{iv}./epsilon_l+ ones(size(Testdata_mod.X_Tst{iv})));
%     
%     Traindata_mod.X_L{iv}(find(Traindata.y_L == 1),:)
%     Traindata_mod.X_L{iv}(find(Traindata.y_L == -1),:)
%     TempXv1 = [];
%     TempXv2 = [];
% end



[accuracy_ct, errorlist_ct, prob_tst_ct, Model_ct,...
    history_ct, history_tst_ct] = mv_cotrainingTWC(Traindata, Testdata, param_ct);

accuracy_ct_array(R,idL) = accuracy_ct*100;  


%% Train AND Test SVM-2K

[acorr,acorr1,acorr2,tpre,tpre1,tpre2,ga,gb,bam,bbm,alpha_A,alpha_B]= ...
mc_svm_2k_lava2(Traindata.X_L{1}, Traindata.X_L{2}, Traindata.y_L, Testdata.X_Tst{1},Testdata.X_Tst{2},Testdata.y_Tst,1,1,1,1e-3,1);


accuracy_svm2k_array(R,idL) = acorr*100;  

%% Train AND Test MV-MED by Sun

[accuracy, errorlist, dec_struct, programflag] = ...
    mvmed_Sun(Traindata, Testdata, param); 

accuracy_mvmedSun_array(R,idL) = accuracy*100; 


   Traindata = [];
   Testdata = [];
end

end
display(sprintf('Accuracy v1: %.1f %% pm %.1f ', mean(accuracy_v1_array(:,idL)), std(accuracy_v1_array(:,idL),[],1)))
display(sprintf('Accuracy v2: %.1f %% pm %.1f ', mean(accuracy_v2_array(:,idL)), std(accuracy_v2_array(:,idL),[],1)))
display(sprintf('Accuracy SVM-2K: %.1f %% pm %.1f', mean(accuracy_svm2k_array(:,idL)), std(accuracy_svm2k_array(:,idL),[],1)))
display(sprintf('Accuracy MV-MED(Sun): %.1f %% pm %.1f', mean(accuracy_mvmedSun_array(:,idL)), std(accuracy_mvmedSun_array(:,idL),[],1)))
display(sprintf('Accuracy CMV-MED: %.1f %% pm %.1f', mean(accuracy_mv_array(:,idL)), std(accuracy_mv_array(:,idL),[],1)))
display(sprintf('Accuracy co-training: %.1f %%  pm %.1f', mean(accuracy_ct_array(:,idL)), std(accuracy_ct_array(:,idL),[],1)))

%Plot 
figure(1)
errorbar(nL_ex, mean(accuracy_v1_array,1), 0.6*std(accuracy_v1_array,[],1), ':r','Linewidth',1.5);
hold on;
errorbar(nL_ex, mean(accuracy_v2_array,1), 0.6*std(accuracy_v2_array,[],1), ':b','Linewidth',1.5);
errorbar(nL_ex, mean(accuracy_svm2k_array,1), 0.5*std(accuracy_svm2k_array,[],1), '-.*g','Linewidth',2);
%errorbar(nL_ex, mean(accuracy_ct_array,1), 0.6*std(accuracy_ct_array,[],1), '-.om','Linewidth',2,'MarkerSize',8);
errorbar(nL_ex, mean(accuracy_mvmedSun_array,1), 0.6*std(accuracy_mvmedSun_array,[],1), '-.xb','Linewidth',2,'MarkerSize',8);
errorbar(nL_ex, mean(accuracy_mv_array,1), 0.6*std(accuracy_mv_array,[],1), '-r','Linewidth',3,'MarkerSize',8);
hold off
grid on
axis([45,255,65,95])%([15,210,65,100])%([5,205,60,100])%([0,250,40,100])
xlabel('number of labeled samples','FontSize',15);
ylabel('accuracy (%)','FontSize',15);
legend('MED v1', 'MED v2', 'SVM-2K','MV-MED', 'CMV-MED','FontSize',18)
%legend('MED v1', 'MED v2', 'SVM-2K','Co-training','MV-MED', 'MV-MED-CIM','FontSize',18)

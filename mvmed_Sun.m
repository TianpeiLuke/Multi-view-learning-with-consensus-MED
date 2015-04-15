function [accuracy, errorlist, dec_struct, programflag] = ...
    mvmed_Sun(Traindata, Testdata, param) 
%% function that learns the multiview binary MED classifier using consistent margin
%  Implementation of MV-MED by Sun and Chao, Multiview Maximum Entropy Discrimination, AAAI 2013 
% Input: 
%   Traindata:  a struct for training set
%            .nV:  no. of views
%            .nU:  no. of unlabeled samples
%            .nL:  no. of labeleed samples
%            .d:   1 x nV, each for dimension of features in one view
%            .X_U: 1 x nV cell structure for unlabeled data
%                  each cell contains a nU x d(i) data 
%            .X_L: 1 x nV cell structure for labeled data
%                  each cell contains a nL x d(i)data set 
%            .y_L: nL x 1 labels 
%            .y_U: nU x 1 the ground truth for unlabeled data (not used in learning)
%   Testdata:  a struct for test set
%            .nTst:  no. of unlabeled samples
%            .d:   dimension of features
%            .X_Tst: 1 x nV cell structure for test data
%                    each cell contains a nTst x d(i) data set
%            .y_Tst: nTst x 1 the ground truth for test data (for error estimate)
%     param:  a struct for parameters
%            .kernelMethod:  
%                     'linear' for linear kernel
%                     'rbf' for Gaussian RBF kernel
%                     'poly' for polynominial kernel
%            .kernelParm:
%                       if 'linear', no need
%                   elseif 'rbf', 
%                       for variance sigma_k
%                   elseif 'poly' 
%                       for degree of polynominal k and bias term b
%            .regParam:     parameter for margin regularization
%            .mode:    =1; for normal mode
%                      =0; no accuracy computed
%   iniparam:    a struct for initial parameters
%            .inimodel: 1 x nV cell for initial model computed 
%
%
% Output:
%   accuracy: expected accuracy under consensus q(y|x1,x2)
%   dev_tst:  nTst x nV+1 decision value for test samples; the last column
%             is from consensus view
%
%  prob_tst:  nTst x nV+1, probability on test samples; the last column is 
%              consensus probability on test samples
% 
%
%   dev_trn:  a struct for decision values
%          .f_trn_u:  nU x nV decision value for unlabeled samples
%          .f_trn_l:  nL x nV decision value for labeled samples
%  prob_trn:  nU x 1, (consensus) probability values on unlabeled
%              samples
%          
%  history:  a struct for tracking the training procedure
%  history_tst: a struct for tracking the testing procedure
%
% Written by Tianpei Xie, Sep 10. 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






addpath('../../../../../../MATLAB/cvx/');

nU = Traindata.nU;
nL = Traindata.nL;
nV = Traindata.nV;
nTst = Testdata.nTst;

programflag  =1;


regParam = 1; %param.regParam;

options = cell(1,nV);
for iv=1:nV
   options{iv} = struct('Kernel', param.kernelMethod{iv}, 'KernelParam', param.kernelParm(iv)); 
end


X_U = Traindata.X_U;
y_U = Traindata.y_U;
X_L = Traindata.X_L;
y_L = Traindata.y_L;

X_Tst = Testdata.X_Tst;
y_Tst = Testdata.y_Tst;

programflag = 1;


% compute the kernel matrix
K_u = cell(1,nV);
K_l = cell(1,nV);
K_ul = cell(1,nV);
K_utst = cell(1,nV);  % kernel for test samples
K_ltst = cell(1,nV);  
K_tst2 = cell(1,nV);
for i=1:nV
    K_u{i} = calckernel(options{i},X_U{i}) + 0.5*ones(nU,1)*ones(nU,1)';
    K_l{i}=  calckernel(options{i},X_L{i}) + 0.5*ones(nL,1)*ones(nL,1)';
    K_ul{i} = calckernel(options{i},X_L{i},X_U{i}) + 0.5*ones(nU,1)*ones(nL,1)'; %U*L'
    K_utst{i} = calckernel(options{i},X_Tst{i},X_U{i})+ 0.5*ones(nU,1)*ones(nTst,1)'; %U*Tst' kernel for test samples
    K_ltst{i} = calckernel(options{i},X_Tst{i},X_L{i})+ 0.5*ones(nL,1)*ones(nTst,1)'; %L*Tst'
    K_tst2{i} = calckernel(options{i},X_Tst{i},X_Tst{i})+ 0.5*ones(nTst,1)*ones(nTst,1)';
end


eones = ones(nL,1);
dual_alpha = zeros(nL,nV);
bias = zeros(1,nV);
%% Make dual programming
 cvx_begin sdp
   %cvx_precision high
   variable alpha1(nL);
   variable alpha2(nL);
   minimize(-eones'*(alpha1+ alpha2+ log(1 - (alpha1+ alpha2)/regParam)) + 0.5*alpha1'*(K_l{1}.*(y_L*y_L'))*alpha1 + 0.5*alpha2'*(K_l{2}.*(y_L*y_L'))*alpha2)
      subject to
         alpha1 >= 0;
         alpha2 >= 0;
  cvx_end 
  
  status = cvx_status;
if strcmp(status,'Solved') || strcmp(status,'Inaccurate/Solved')
  dual_alpha(:,1) = alpha1;
  dual_alpha(:,2) = alpha2;
%   idx_sv1 = find(alpha1~=0);
%   idx_sv2 = find(alpha2~=0);
%   bias(1) = sum(1./y_L(idx_sv1) - K_l{1}(idx_sv1,:)*alpha1);
%   bias(2) = sum(1./y_L(idx_sv2) - K_l{2}(idx_sv2,:)*alpha2);
  dec_struct.mean = zeros(nTst, nV);
  dec_struct.dualParm = dual_alpha;
%%  Prdiction  
  dec_struct.mean(:,1) = K_ltst{1}'*(alpha1.*y_L) + bias(1);
  dec_struct.mean(:,2) = K_ltst{2}'*(alpha2.*y_L) + bias(2);
  dec_struct.mean(:,3) = 0.5*dec_struct.mean(:,1) + 0.5*dec_struct.mean(:,2);
  
  errorlist = (sign(dec_struct.mean(:,3))~=y_Tst);
  accuracy_comb = 1 - sum(errorlist)/nTst;

  errorlist1 = (sign(dec_struct.mean(:,1))~=y_Tst);
  accuracy_v1 = 1 - sum(errorlist1)/nTst;
  
  errorlist2 = (sign(dec_struct.mean(:,2))~=y_Tst);
  accuracy_v2 = 1 - sum(errorlist2)/nTst;

  accuracy = max([accuracy_comb,accuracy_v1,accuracy_v2]);
  
  display(sprintf('Accuracy: %.2f %%', accuracy*100));
  
  
  else
      display('Error: no solution')
      programflag = 0;
      accuracy = 0; 
      errorlist = []; 
      dec_struct.mean = [];
        
%        hisory.vb_history = vb_history;
%         hisory.biasmap_history= biasmap_history;
%         hisory.biasjoint_history = biasjoint_history;
        history_tst = [];
      return;
   end
end




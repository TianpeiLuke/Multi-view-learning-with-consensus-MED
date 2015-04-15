function [accuracy, errorlist, dec_struct, prob_tst, dev_trn, prob_trn, ...
           history, history_tst,programflag] = cmvmed(Traindata, Testdata, param, iniparam) 
%% function that learns the multiview binary MED classifier using MED and complementary information minimization
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
%            .regParam:     parameter for B-distance regularization
%            .maxIterOut:  maximum iteration for the outer loop
%            .threOut   :  stopping threshold for the outer loop
%            .maxIterMAP:  maximum iteration for MAP computing
%            .threMAP:     stopping threshold for MAP computing
%            .sigmaPri:    sigma for prior of weights
%            .q_thresh:    threshold for select most confidence data
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
% Written by Tianpei Xie, Feb 25. 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../../../../../MATLAB/cvx/');
addpath('./ann_mwrapper')
nU = Traindata.nU;
nL = Traindata.nL;
nV = Traindata.nV;
nTst = Testdata.nTst;

maxIter = param.maxIterOut;
sigma2 = (param.sigmaPri)^2;
epsilon = param.threMAP;
q_thresh = param.q_thresh;
model = iniparam.inimodel;

regParamUpper = param.regParam;

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

% for iv=1:nV
%   if strcmp(options{iv}.Kernel, 'linear')
%       X_U{iv} = [X_U{iv}, 5*ones(size(X_U{iv},1),1)/nU]; %augmented 1-dim
%       X_L{iv} = [X_L{iv}, 5*ones(size(X_L{iv},1),1)/nL];
%       X_Tst{iv}= [X_Tst{iv}, 5*ones(size(X_Tst{iv},1),1)/nTst];
%   end
% end

% compute the kernel matrix
K_u = cell(1,nV);
K_l = cell(1,nV);
K_ul = cell(1,nV);
K_utst = cell(1,nV);  % kernel for test samples
K_ltst = cell(1,nV);  
K_tst2 = cell(1,nV);
for i=1:nV
    K_u{i} = calckernel(options{i},X_U{i});
    K_l{i}=  calckernel(options{i},X_L{i});
    K_ul{i} = calckernel(options{i},X_L{i},X_U{i}); %U*L'
    K_utst{i} = calckernel(options{i},X_Tst{i},X_U{i}); %U*Tst' kernel for test samples
    K_ltst{i} = calckernel(options{i},X_Tst{i},X_L{i}); %L*Tst'
    K_tst2{i} = calckernel(options{i},X_Tst{i},X_Tst{i});
end
sigmoid = @(x)(1./(1+ exp(-x))); % classifier for binary set

q = 0.5*ones(nU,1);  % consensus probablity on unlabeled set
proj_p = 0.5*ones(nU,nV); %pseudo-labling prob for each view on unlabeled set
q_tst = 0.5*ones(nTst, 1); 
proj_p_tst = 0.5*ones(nTst,nV);
errorlist = zeros(nTst, 1);
accuracy = 0;

fl0 = zeros(nL,nV);  % initial decision value on labeled set
fu0 = zeros(nU,nV);  %                   ...  on unlabeled set
ftst0 = zeros(nTst,nV); % initial decision value on test set
% bias0 = zeros(1,nV);

B = cell(1,nV);
B_tst = cell(1,nV);

dec_struct.mean = zeros(nTst,nV+1);
dec_struct.var  = zeros(nTst,nTst,nV);
prob_tst = zeros(nTst,nV+1);
%%----------------------- history tracking---------------------
v_history = zeros(nU,nV,maxIter+1);
% vb_history = zeros(nV, maxIter+1);
proj_history = zeros(nU, nV, maxIter+1);
q_history = zeros(nU, maxIter+1);
proj_v_history = zeros(nU, nV,maxIter+1);
dev_history = zeros(nU,nV, maxIter+1);
dual_history = zeros(nL,nV, maxIter+1);
p_history = zeros(nU,nV, maxIter+1);
fmap_history = zeros(nU, nV, maxIter+1);
fpred_history = zeros(nL, nV, maxIter+1);
fjointu_history = zeros(nU, nV, maxIter+1);
fjointl_history = zeros(nL, nV, maxIter+1);

% biasmap_history = zeros(nV, maxIter+1);
% biasjoint_history = zeros(nV, maxIter+1);

fmap_tst_history= zeros(nTst, nV, maxIter+1);
fjoint_tst_history= zeros(nTst, nV, maxIter+1);
q_tst_history = zeros(nTst, maxIter+1);
%% -------------------- Initialization ----------------------
 % compute the initial value of SVM decision
display(sprintf('\n============================================'));
display(sprintf('Initializing...'));
X_L_temp =cell(1,nV); X_U_temp  =cell(1,nV); X_Tst_temp=cell(1,nV);

for i=1:nV    
   if strcmp(options{i}.Kernel, 'chisquare')
      indSV = full(model{i}.SVs);    
      Kl_temp = calckernel(options{i},X_L{i}(indSV,:),X_L{i});
      
     Ku_temp = calckernel(options{i},X_L{i}(indSV,:),X_U{i});
   
     Ktst_temp = calckernel(options{i},X_L{i}(indSV,:),X_Tst{i});
           fl0(:,i) = model{iv}.decision_valL;%Kl_temp*model{i}.sv_coef;
      fu0(:,i) = model{iv}.decision_valU; %Ku_temp*model{i}.sv_coef;
      ftst0(:,i) = model{iv}.decision_valTst; %Ktst_temp*model{i}.sv_coef;
  else  
     Kl_temp = calckernel(options{i},full(model{i}.SVs),X_L{i});
     Ku_temp = calckernel(options{i},full(model{i}.SVs),X_U{i});
    % SVM Prediction
     Ktst_temp = calckernel(options{i},full(model{i}.SVs),X_Tst{i});
      fl0(:,i) = Kl_temp*model{i}.sv_coef -model{i}.rho;
      fu0(:,i) = Ku_temp*model{i}.sv_coef  -model{i}.rho;
      ftst0(:,i) = Ktst_temp*model{i}.sv_coef -model{i}.rho;
  end
end

view_weight = 0.5*ones(nV,1);

% ------------------------- view combination -----------------------------
q= consensus_comp(fu0,view_weight);
q_tst= consensus_comp(ftst0,view_weight); 


% ------------------------- view projection ----------------------------
% find nearest neigbor graph
k = 8; nnidxU = cell(1,nV); IdxMatU = cell(1,nV); 
nnidxTst = cell(1,nV); IdxMatTst= cell(1,nV);
for i=1:nV
nnidxU{i} = annquery(X_U{i}', X_U{i}', k);
IdxMatU{i}  = nnidxU{i}';

nnidxTst{i} = annquery(X_U{i}', X_Tst{i}', k);
IdxMatTst{i}  = nnidxTst{i}';
end
% find the projected density in each view on Delta(y)
for i=1:nV
 [B{i}, proj_p(:,i), ~, ~] = weightopt(q, fu0(:,i), IdxMatU{i}, 5e-3);
 [B_tst{i}, proj_p_tst(:,i), ~, ~] = weightopt(q, ftst0(:,i), IdxMatTst{i}, 5e-3);
end



q_history(:,1) = q;
dev_history(:,:,1) = fu0;
q_tst_history(:,1) = q_tst;
for i=1:nV
  proj_v_history(:,i,1) = proj_p(:,i);
end
%%
 % find the MAP estimate for unlabeled part of decision
fmap = fu0;  %map estimate on unlabeled data
fpred = fl0; %map estimate on labeled data
ftst_map = ftst0;   %map estimate on test data



fjointu = fmap;
fjointl = fpred;
fjoint_tst = ftst_map;

% update M for Hessian matrix 
v = zeros(nU, nV);
M = zeros(nU, nU, nV);

for i=1:nV
    v(:,i) = 1/16*sech(fmap(:,i)).^2; %1/2*1./(1+cosh(fmap(:,i)));
    M(:,:,i) = diag(v(:,i));
end
v_history(:,:,1) = v;


p =  0.5*ones(nU,nV);
p_history(:,:,1) = p;
fmap_history(:,:,1) = fmap;
fpred_history(:,:,1) = fpred;
fjointu_history(:,:,1) = fjointu;
fjointl_history(:,:,1) = fjointl;

fmap_tst_history(:,:,1)= ftst0;
fjoint_tst_history(:,:,1) = fjoint_tst;

%track the probablity 
%track_diff = zeros(nV,param.maxIterMAP);
 history.mapupdate = cell(1, maxIter); %cell(param.maxIterMAP, maxIter);
 history.qfilter = cell(nV,maxIter);
 history.indfilter = cell(nV,maxIter);
%% ------------ outer loop for EM-style learning ---------------
iout = 0;
outflag = 1;
rho = 1.1+1.5*rand(1);
while( outflag && iout< maxIter)
 iout = iout+1;
 display(sprintf('outer loop: i=%d \n =============================================',iout));
 q_pre = q;
 proj_p_pre = proj_p;
 
 regParam = (1 - exp(-0.5*iout))*regParamUpper; %(1 - exp(-0.5*1)); 

 
 t= 0;
 flag_w1 = 1;
 dual_alpha = zeros(nL, nV);
 dual_eta = zeros(nU, nV);

 %initalization of fmap
  if iout ==1
   fmap = fu0;  %map estimate on unlabeled data
   fpred = fl0; %map estimate on labeled data
   ftst_map = ftst0;
  else 
   fmap= fjointu;                       
    %lableled part
   fpred =fjointl; 
    %Prediction  
   ftst_map = fjoint_tst;
  end

 
    %epsilon_t =  5e-4;
    for i=1:nV
        p(:,i) = sigmoid(fmap(:,i));
        if strcmp(options{i}.Kernel, 'linear')
         p(:,i) = sigmoid(fmap(:,i)- model{i}.rho);  
        end
       errorlistp = (sign(p(:,i) - 0.5*ones(nU,1))~= y_U) ; 
       accuracyp(i) = 1 - sum(errorlistp)/nU;
       errorlistp= [];
    end
    %
    view_weight = 1/nV*ones(nV,1);
    qt = consensus_comp(fmap,view_weight);
    if strcmp(options{i}.Kernel, 'linear')
      qt =consensus_comp(bsxfun(@minus, fmap, [model{1}.rho,model{2}.rho]),view_weight);
    end
    errorlistp = (sign(qt - 0.5*ones(nU,1))~= y_U) ;
    accuracyp(3) = 1 - sum(errorlistp)/nU;
    errorlistp= [];
    history.mapupdate{iout} = accuracyp;
    
  fmap_history(:,:,iout+1) = fmap;
  fpred_history(:,:,iout+1) = fpred;
  fmap_tst_history(:,:,iout+1)= ftst_map;
  

  %%
  for i=1:nV
     p(:,i) =  sigmoid(fmap(:,i));
  end
  p_history(:,:,iout+1) = p;
  
  for i=1:nV
    v(:,i) = 1/16*sech(fmap(:,i)).^2; %1/2*1./(1+cosh(fmap(:,i)));
    M(:,:,i) = diag(v(:,i));
  end
  v_history(:,:,iout+1) = v;

%% Filtering data
%if iout >= 1
for jj=1:nV
 isupper = 1; islower = 1; 
 while(isupper && islower)
 ind_q_clr = union(find(proj_p(:,jj) >= q_thresh), find(1-proj_p(:,jj) >= q_thresh));
 ind_q_cor = setdiff(1:length(proj_p(:,jj)), ind_q_clr);
    if isempty(ind_q_clr)
     q_thresh = max([q_thresh - 1/3*(q_thresh - 0.5-1e-3), 0.5+1e-3]);
    elseif isempty(ind_q_cor)
     q_thresh = min([q_thresh + 1/3*(1- 1e-3 - q_thresh), 1-1e-3]);
    else
       isupper = 0;
       islower = 0;
    end
 end
 y_ext(:,jj) = [y_L; sign(proj_p(ind_q_clr,jj) - 0.5*ones(length(ind_q_clr),1))];
 y_psudo(:,jj) = sign(proj_p(ind_q_clr,jj) - 0.5*ones(length(ind_q_clr),1)); %pseudolabel
  history.qfilter{jj,iout} = max([proj_p(ind_q_clr,jj), 1-proj_p(ind_q_clr,jj)], [],2);
  history.indfilter{jj,iout} = ind_q_clr;
end  
 P = zeros(nL+length(ind_q_clr), nL+length(ind_q_clr),  nV);
 U = zeros(nL+length(ind_q_clr), nL+length(ind_q_clr),  nV);
 S = zeros(nL+length(ind_q_clr), nL+length(ind_q_clr),  nV);
 
 b = [ones(nL,nV); zeros(length(ind_q_clr),nV)]; 
% else
%  P = zeros(nL, nL,  nV);
%  U = zeros(nL, nL,  nV);
%  S = zeros(nL, nL,  nV);  
%  
%   b = ones(nL,nV);  
% end
  %% find sparse dual variables for labeled part 
 param_qu.sigma2 = sigma2 ;
 for ii=1:nV     
        K_LU1_temp =[];
        K_LU1_temp = [K_l{ii}, regParam*K_ul{ii}(ind_q_clr,:)';regParam*K_ul{ii}(ind_q_clr,:),  regParam^2*K_u{ii}(ind_q_clr,ind_q_clr)];
        K_U2LU1_temp =[];
        K_U2LU1_temp = [K_ul{ii}(ind_q_cor,:), regParam*K_u{ii}(ind_q_cor,ind_q_clr)];
 
        P(:,:,ii) =  (rho*K_LU1_temp - regParam*(K_U2LU1_temp'*((pinv(M(ind_q_cor,ind_q_cor,ii))/sigma2 +...
                               regParam*K_u{ii}(ind_q_cor,ind_q_cor))\K_U2LU1_temp))).*(y_ext*y_ext');                           
%      else
%         P(:,:,ii) =  K_l{ii}; %- regParam*(K_ul{ii}'*((pinv(M(:,:,ii))/sigma2 +...
%                              %regParam*K_u{ii})\K_ul{ii}))).*(y_L*y_L');
%      end
  
  [U(:,:,ii), S(:,:,ii)] = eig(P(:,:,ii));
  % call qudratic-solver to solve it
%  if iout>= 1
    [alpha, omega, status] = quadr_svm_ext(0.5*(P(:,:,ii)+P(:,:,ii)'), b(:,ii), max([q(ind_q_clr), 1-q(ind_q_clr)],[],2), nL, length(ind_q_clr),param_qu);
%   else
%     [alpha, status] = quadr_svm(0.5*(P(:,:,ii)+P(:,:,ii)'), b(:,ii), param_qu);
%   end
%   
  if strcmp(status,'Solved') || strcmp(status,'Inaccurate/Solved')
      dual_alpha(:,ii) = alpha;
      if iout>= 1
        dual_eta(ind_q_clr,ii) = omega;
      end
  else
      display('Error: no solution')
      programflag = 0;
      accuracy = 0; 
      errorlist = []; 
      dec_struct.mean = []; 
      prob_tst = []; 
      dev_trn = []; 
      prob_trn = []; 
      history.v_history = v_history;
      history.q_history = q_history;
        history.dev_history = dev_history;
        history.dual_history = dual_history;
        history.p_history = p_history;
        history.fmap_history = fmap_history;
        history.fpred_history = fpred_history;
        history.fjointu_history = fjointu_history;
        history.fjointl_history = fjointl_history;

        history_tst = [];
      return;
  end
 end %% End of Sparse coefficient learning
 

%%
for i=1:nV
    %unlabeled part
    %if iout >= 1
    fjointu(:,i) =  sigma2*K_ul{i}*(y_L.*dual_alpha(:,i)) ...
                    + sigma2*regParam*K_u{i}(:,ind_q_clr)*(y_psudo(:,jj).*dual_eta(ind_q_clr,i))...
                    - sigma2*regParam*K_u{i}(:,ind_q_cor)*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_ul{i}(ind_q_cor,:))*(y_L.*dual_alpha(:,i))...
                    - sigma2*regParam^2*K_u{i}(:,ind_q_cor)*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_u{i}(ind_q_cor,ind_q_clr))*(y_psudo(:,jj).*dual_eta(ind_q_clr,i));
                         
    %lableled part
    fjointl(:,i) =  sigma2*K_l{i}*(y_L.*dual_alpha(:,i)) ...
                    + sigma2*regParam*K_ul{i}(ind_q_clr,:)'*(y_psudo(:,jj).*dual_eta(ind_q_clr,i))...                       
                    - sigma2*regParam*K_ul{i}(ind_q_cor,:)'*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_ul{i}(ind_q_cor,:))*(y_L.*dual_alpha(:,i))...
                    - sigma2*regParam^2*K_ul{i}(ind_q_cor,:)'*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_u{i}(ind_q_cor,ind_q_clr))*(y_psudo(:,jj).*dual_eta(ind_q_clr,i));
    
    %Prediction  
    fjoint_tst(:,i) = sigma2*K_ltst{i}'*(y_L.*dual_alpha(:,i)) ...
                    + sigma2*regParam*K_utst{i}(ind_q_clr,:)'*(y_psudo(:,jj).*dual_eta(ind_q_clr,i))...                       
                    - sigma2*regParam*K_utst{i}(ind_q_cor,:)'*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_ul{i}(ind_q_cor,:))*(y_L.*dual_alpha(:,i))...
                    - sigma2*regParam^2*K_utst{i}(ind_q_cor,:)'*...
                             ((pinv(M(ind_q_cor,ind_q_cor,i))/sigma2 + regParam*K_u{i}(ind_q_cor,ind_q_cor))...
                             \K_u{i}(ind_q_cor,ind_q_clr))*(y_psudo(:,jj).*dual_eta(ind_q_clr,i));
%     else
%         fjointu(:,i) =  sigma2*K_ul{i}*(y_L.*dual_alpha(:,i));
%         fjointl(:,i) =  sigma2*K_l{i}*(y_L.*dual_alpha(:,i));
%         fjoint_tst(:,i) = sigma2*K_ltst{i}'*(y_L.*dual_alpha(:,i));
%     end
    %unlabeled part 
%     fjointu(:,i) =  fmap(:,i) + sigma2*K_ul{i}*(y_L.*dual_alpha(:,i)) ...
%                              - sigma2*regParam*K_u{i}*...
%                              ((pinv(M(:,:,i))/sigma2 + regParam*K_u{i})...
%                              \K_ul{i})*(y_L.*dual_alpha(:,i));
                         
    %lableled part
%     fjointl(:,i) =  fpred(:,i)+ sigma2*K_l{i}*(y_L.*dual_alpha(:,i)) ...
%                              - sigma2*regParam*K_ul{i}'*...
%                              ((pinv(M(:,:,i))/sigma2 + regParam*K_u{i})...
%                              \K_ul{i})*(y_L.*dual_alpha(:,i));
    
    %Prediction  
%     fjoint_tst(:,i) =  ftst_map(:,i)+ sigma2*K_ltst{i}'*(y_L.*dual_alpha(:,i)) ...
%                              - sigma2*regParam*K_utst{i}'*((pinv(M(:,:,i))/sigma2 ...
%                              + regParam*K_u{i})\K_ul{i})*(y_L.*dual_alpha(:,i));
end

fjointu_history(:,:,iout+1) = fjointu;
fjointl_history(:,:,iout+1) = fjointl;
fjoint_tst_history(:,:,iout+1) = fjoint_tst;
% 
%  if strcmp(options{i}.Kernel, 'linear')
%     biasjoint_history(:,iout+1)= bias_jount';
%  end
%% new consensus view
view_weight = 1/nV*ones(nV,1);

q= consensus_comp(fjointu,view_weight); %compute the averge prediction 
q_tst = consensus_comp(fjoint_tst,view_weight); 
if strcmp(options{i}.Kernel, 'linear')
 q= consensus_comp(bsxfun(@minus, fjointu, [model{1}.rho,model{2}.rho]),  view_weight); %compute the averge prediction 
 q_tst = consensus_comp(bsxfun(@minus, fjoint_tst, [model{1}.rho,model{2}.rho]),  view_weight); 
end

% find the projected density in each view on Delta(y)
for i=1:nV
 [B{i}, proj_p(:,i), ~, ~] = weightopt(q, fjointu(:,i), IdxMatU{i}, 5e-3);
 [B_tst{i}, proj_p_tst(:,i), ~, ~] = weightopt(q, fjoint_tst(:,i), IdxMatTst{i}, 5e-3);
end



q_history(:,iout+1) = q; 
q_tst_history(:,iout+1) = q_tst;
for i=1:nV
  proj_v_history(:,i,iout+1) = proj_p(:,i);
end

dual_history(:,:,iout) =  dual_alpha;
dual_eta_history(:,:,iout) =  dual_eta;

if norm(q - q_pre)/nU < param.threOut
    outflag = 0;
    q_tst_history(:,iout+2:end) = [];
    proj_v_history(:,:,iout+2:end) = [];
    v_history(:,:,iout+2:end) = [];
    dev_history(:,:,iout+2:end)= [];
    dual_history(:,:,iout+1:end) = [];
    p_history(:,:,iout+2:end)= [];
    fmap_history(:,:,iout+2:end)= [];
    fpred_history(:,:,iout+2:end)= [];
    fjointu_history(:,:,iout+2:end)= [];
    fjointl_history(:,:,iout+2:end)= [];


    q_tst_history(:,iout+2:end)= [];
    fmap_tst_history(:,:,iout+2:end) = [];
    fjoint_tst_history(:,:,iout+2:end) = [];
     history.mapupdate(:,iout+1:end) = [];
end
    


end %% End of outer loop
display('End of iteration');
%% Prediction
display('Prediction ...')
dec_struct.mean(:,1:nV) = fjoint_tst;
dec_struct.mean(:,nV+1) = fjoint_tst*view_weight;



% meanfun = zeros(nTst, nV);
% avg_pred = zeros(nTst, nV);
% var_pred = zeros(nTst, nV);
% label_pred = zeros(nTst, nV);
% varfun = zeros(nTst,nTst, nV);
% for iv = 1:nV
%    % mean of functions
%    meanfun(:,iv) = fjoint_tst(:,iv);
%    % variance of functions
%    varfun(:,:,iv) = rho*sigma2*K_tst2{iv} - sigma2*K_utst{iv}'*((pinv(M(:,:,iv))/sigma2 ...
%                              + K_u{i})\K_utst{iv});
%                          
%    [Us,Ds] = eig(0.5*(varfun(:,:,iv)+varfun(:,:,iv)'));
%    Ds = real(Ds); % elimate the case when numerical issue happens
%    ss= diag(Ds);
%    ss(find(ss < 0)) = zeros(length(find(ss < 0)),1);                      
%   dec_struct.var(:,:,iv) =  Us*Ds*Us';                  
%  [avg_pred(:,iv), var_pred(:,iv)] = SampleSigmoid(meanfun(:,iv)', 0.5*(varfun(:,:,iv)+varfun(:,:,iv)'));
%  label_pred(:,iv) = sign(avg_pred(:,iv) - 0.5*ones(nTst,1));
% end


if param.mode == 1
   dec  =zeros(nTst, 1); 
   dec  = sign(q_tst - 0.5*ones(nTst,1));
  for i=1:nV
      dec = dec + sign(proj_p_tst(:,i) - 0.5*ones(nTst,1));
  end
  errorlist = (sign(dec)~= y_Tst) ; %(sign(q_tst - 0.5*ones(nTst,1))~= y_Tst) ; 
  %errorlist = (sign(label_pred*view_weight)~= y_Tst) ; 
  accuracy = 1 - sum(errorlist)/nTst;
  display(sprintf('Accuracy: %.2f %%', accuracy*100));
elseif param.mode == 0
    errorlist = [];
    accuracy  = -1;
    display('Prediction Ends')
end

%%
dev_trn.f_trn_u = fjointu;
dev_trn.f_trn_l = fjointl;
       prob_trn = q;
       
for ivv = 1:nV
   prob_tst(:,ivv) = sigmoid(fjoint_tst(:,ivv)); 
%     if strcmp(options{ivv}.Kernel, 'linear')
%        prob_tst(:,ivv) = sigmoid(fjoint_tst(:,ivv)-model{ivv}.rho); 
%     end
end
prob_tst(:,nV+1) = q_tst;

%% History store
history.v_history = v_history;
history.q_history = q_history;
history.dev_history = dev_history;
history.proj_v_history = proj_v_history;
history.p_history = p_history;
history.fmap_history = fmap_history;
history.fpred_history = fpred_history;
history.fjointu_history = fjointu_history;
history.fjointl_history = fjointl_history;

if maxIter>0
    history.dual_history = dual_history;
history.dual_eta_history = dual_eta_history;
end
history_tst.q_tst_history = q_tst_history;
history_tst.fmap_tst_history = fmap_tst_history;
history_tst.fjoint_tst_history = fjoint_tst_history;
display(sprintf('============================================\n'))
end


%% Auxilary functions
function [alpha, omega, status] = quadr_svm_ext(P, b, q, nL, nU1, param)
%% Auxilary function to compute the quadratic programming of extensive SVM
% Input:   
%        P: n x n positive semidefinite matrix 
%        b: n x 1 linear term
% Output: 
%       alpha: n x 1 dual variables
% 
%
addpath('../../../../../../MATLAB/cvx/');
 n = size(P,1);
 [U,S] = eig(P);
S = real(S); % elimate the case when numerical issue happens
ss= diag(S);
ss(find(ss < 0)) = zeros(length(find(ss < 0)),1);
S = diag(ss); % enforcing to be PSD

sigma2 = param.sigma2;

 cvx_begin sdp
   %cvx_precision high
   variable alpha(nL) ;%nonnegative;
   variable omega(nU1) ;%nonnegative;
   minimize(-b'*[alpha;omega] + 0.5*sigma2*[alpha;omega]'*(U*S*U')*[alpha;omega]  )
     subject to
       alpha <= 1;
       alpha >= 0;
       omega >= 0;
       omega <= q;
  cvx_end 
  
  status = cvx_status;
end


function [alpha, status] = quadr_svm(P, b, param)
%% Auxilary function to compute the quadratic programming of extensive SVM
% Input:   
%        P: n x n positive semidefinite matrix 
%        b: n x 1 linear term
% Output: 
%       alpha: n x 1 dual variables
% 
%
addpath('../../../../../../MATLAB/cvx/');
 n = size(P,1);
 [U,S] = eig(P);
%  if norm(P - P')>1e-5%~isreal(diag(S))
%      error('Error: P must be symmetric');
%  elseif sum(find(diag(S) < 0))>0
%      error('Error: P must be positive semidefinite matrix');
%  end

S = real(S); % elimate the case when numerical issue happens
ss= diag(S);
ss(find(ss < 0)) = zeros(length(find(ss < 0)),1);
S = diag(ss); % enforcing to be PSD

sigma2 = param.sigma2;
eones = ones(n,1);

 cvx_begin sdp
   %cvx_precision high
   variable alpha(n) ;%nonnegative;
   minimize(-b'*alpha + 0.5*sigma2*alpha'*(U*S*U')*alpha )
     subject to
       alpha <= 1;
       alpha >= 0;
  cvx_end 
  
  status = cvx_status;
end




function K = calckernel(options,X1,X2)
% {calckernel} computes the Gram matrix of a specified kernel function.
% 
%      K = calckernel(options,X1)
%      K = calckernel(options,X1,X2)
%
%      options: a structure with the following fields
%               options.Kernel: 'linear' | 'poly' | 'rbf' 
%               options.KernelParam: specifies parameters for the kernel 
%                                    functions, i.e. degree for 'poly'; 
%                                    sigma for 'rbf'; can be ignored for 
%                                    linear kernel 
%      X1: N-by-D data matrix of N D-dimensional examples
%      X2: (it is optional) M-by-D data matrix of M D-dimensional examples
% 
%      K: N-by-N (if X2 is not specified) or M-by-N (if X2 is specified)
%         Gram matrix
%
% Author: Stefano Melacci (2009)
%         mela@dii.unisi.it
%         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

kernel_type=options.Kernel;
kernel_param=options.KernelParam;

n1=size(X1,1);
if nargin>2
    n2=size(X2,1);
end

 switch kernel_type

    case 'linear'
        if nargin>2
            K=X2*X1';
        else
            K=X1*X1';
        end

    case 'poly'
        if nargin>2
            K=(X2*X1').^kernel_param;
        else
            K=(X1*X1').^kernel_param;
        end

    case 'rbf'
        if nargin>2
            K = exp(-(repmat(sum(X1.*X1,2)',n2,1) + ...
                repmat(sum(X2.*X2,2),1,n1) - 2*X2*X1') ...
                /(2*kernel_param^2));
        else
            P=sum(X1.*X1,2);
            K = exp(-(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1')/(2*kernel_param^2));
        end

    case 'chisquare'   
        display('Histogram kernel compuatation')
        if nargin>2
            
           K = pdist2(X2,X1, @chi_square_statistics_fast); 
%            n = size(X1,1); m = size(X2,1);
%            for ii =1:m
%                 K(ii,:) = histogram_intersection(X2(ii,:),X1)';
%            end
         else
%             n = size(X1,1);
%            for ii =1:n
%                 K(ii,:) = histogram_intersection(X1(ii,:),X1)';
%            end
           K = pdist2(X1,X1, @chi_square_statistics_fast); 
         end
        
    otherwise
        
       error('Unknown kernel function.');
 end
end



function q = consensus_comp(f, view_weight)
  nV = size(f,2);

  sigmoid = @(x)(1./(1+ exp(-x)));
   pp = sigmoid(f);
   pn = bsxfun(@minus, ones(1,nV), pp);
  %q = sigmoid(f*view_weight);
   qp = exp(log(pp)*view_weight);
   qn = exp(log(pn)*view_weight);
   q =  bsxfun(@rdivide, exp(log(pp)*view_weight),qp+qn);
end



function [avg_pred, var_pred] = SampleSigmoid(Mu, Sigma)
TotalN = 10000;
sigmoid = @(x)(1./(1+ exp(-x)));
pred_array= zeros(length(Mu),TotalN);
[Us,Ds] = eig(Sigma);
Ds = real(Ds); % elimate the case when numerical issue happens
ss= diag(Ds);
ss(find(ss < 0)) = zeros(length(find(ss < 0)),1);
Ds = diag(ss); % enforcing to be PSD
display('Prediction via sampling')
RandInput = mvnrnd(repmat(Mu, TotalN, 1), (Us*Ds*Us'))';
for n=1:TotalN
  pred_array(:,n) = sigmoid(RandInput(:,n));
end
avg_pred = mean(pred_array,2);
var_pred = std(pred_array, [], 2);
end


function [acorr,acorr1,acorr2,tpre,tpre1,tpre2,ga,gb,bam,bbm,alpha_A,alpha_B]= ...
mc_svm_2k_lava2(XTrain1,XTrain2,YTrain,XTest1,XTest2,YTest,CA,CB,D,eps,ifeature)
% function to call  optimisation function and compute the prediction for the test set
% input:
%               XTrain1     training features for the first SVM matrix, the rows contain the observations
%               XTrain2     training features for the second SVM matrix, the rows contain the observations
%               YTrain      training labels vector
%               XTest1      test features for the first SVM matrix, the rows contain the observations
%               XTest2      test features for the second SVM matrix, the rows contain the observations
%               YTest       test labels vector
%               CA          penalty value for the first SVM
%               CB          penalty value for the second SVM
%               D           penalty for the synthesis
%               eps         tolerance for the synthesis
%
% output:
%               pre         combained prediction for the test
%               pre1        prediction for the test by the first SVM
%               pre2        prediction for the test by the second SVM
%               tpre        combained prediction for the training
%               tpre1       prediction for the training by the first SVM
%               tpre2       prediction for the training by the second SVM
%               ga     dual vector corresponding to the first SVM
%               gb     dual vector corresponding to the second SVM
% 
                ikernel=1;

                [m,n1]=size(XTrain1);
                [m,n2]=size(XTrain2);
                e1=ones(m,1);
                
                mtrain=m;
                
                mtest=max(size(YTest));
                
%                ifeature=1;

                xsigma=0.02;
                ikernel1=3;% 1;
                ikernel2=3; %1;
                idegree1=2;
                idegree2=3;
                if ifeature==1
                  switch ikernel1 
                    case 0
                        KA=XTrain1*XTrain1';  % linear kernel
                    case 1 % polynomial
                        KA=(XTrain1*XTrain1'+1).^idegree1;
                    case 2 % sigmoid
                        KA=tanh(XTrain1*XTrain1'+1);
                    case 3  % Gaussian
                        KA=XTrain1*XTrain1';
                        sigma=xsigma*n1;
                        dd=diag(KA);
                        KA=dd*e1'+e1*dd'-2*KA;
                        KA=exp(-KA/sigma);

                  end
                  switch ikernel2 
                    case 0
                        KB=XTrain2*XTrain2';
                    case 1 % polynomial
                        KB=(XTrain2*XTrain2'+1).^idegree2;
                    case 2 % sigmoid
                        KB=tanh(XTrain2*XTrain2'+1);
                    case 3  % Gaussian
                        sigma=xsigma*n2;
                        KB=XTrain2*XTrain2';
                        dd=diag(KB);
                        KB=dd*e1'+e1*dd'-2*KB;
                        KB=exp(-KB/sigma);
                  end
                else
                  KA=XTrain1;
                  KB=XTrain2;
                end


% penalty parameters                
%                CA=0.2;  % SVM1
%                CB=0.2;  % SVM2
%                D=0.1;   % interaction
%                eps=0.001;
% solver 
                tic;
                if ifeature==1 && ikernel==0
                    [alpha_A,alpha_B,u_N]=mc_svm_2k_alcg_ls2(XTrain1,XTrain2,YTrain,CA,CB,D,eps,ikernel,ifeature);                
                else
                    [alpha_A,alpha_B,u_N]=mc_svm_2k_alcg_ls2(KA,KB,YTrain,CA,CB,D,eps,ikernel,0);                
                end    
                xtime=toc;
                ga=YTrain.*alpha_A-u_N;
                gb=YTrain.*alpha_B+u_N;
                
                
                indp=find(YTrain==1);
                indn=find(YTrain==-1);
                
                epsC=0.005;
                min_alpha=0;
                max_alpha=CA;
                ind0=(alpha_A>epsC+min_alpha);
                ind1=(alpha_A<max_alpha-epsC);
                inxa=find(ind0.*ind1);
% bias for problem A         

                
                ba=mean(YTrain(inxa)-KA(inxa,:)*ga);
                
                min_alpha=0;
                max_alpha=CB;
                ind0=(alpha_B>epsC+min_alpha);
                ind1=(alpha_B<max_alpha-epsC);
                inxb=find(ind0.*ind1);
% bias for problem B
                bb=mean(YTrain(inxb)-KB(inxb,:)*gb);
                
                sxA=sort(ga);
                sxB=sort(gb);
                
                ytrsum=sum(YTrain);
                isxA=(mtrain-ytrsum)/2;
                
                ba=(sxA(isxA)+sxA(isxA+1))/2;
                bb=(sxB(isxA)+sxB(isxA+1))/2;
                
%                ba=0;
%                bb=0;
                
% prediction on the training 
                gaKA=ga'*KA;
                gbKB=gb'*KB;
                tpre1=(gaKA+ba)';
                tpre2=(gbKB+bb)';
                
                
                thit1=YTrain.*sign(tpre1);
                thit2=YTrain.*sign(tpre2);
                
                
                tpre=(tpre1+tpre2)/2;
                thit=YTrain.*sign(tpre);
                tacorr=(sum(thit)+mtrain)/(2*mtrain);               
                tacorr1=(sum(thit1)+mtrain)/(2*mtrain);
                tacorr2=(sum(thit2)+mtrain)/(2*mtrain);

                bam=mc_adjust_bias(ba,tpre1,gaKA,YTrain);
                bbm=mc_adjust_bias(bb,tpre2,gbKB,YTrain);
                
% prediction on the training by the new biases                
                tpre1=(gaKA+bam)';
                tpre2=(gbKB+bbm)';
                thit1=YTrain.*sign(tpre1);
                thit2=YTrain.*sign(tpre2);

                xlambda=0.5;
                
                
                tpre=xlambda*tpre1+(1-xlambda)*tpre2;
                thit=YTrain.*sign(tpre);
                tacorr=(sum(thit)+mtrain)/(2*mtrain);               
                tacorr1=(sum(thit1)+mtrain)/(2*mtrain);
                tacorr2=(sum(thit2)+mtrain)/(2*mtrain);

% prediction on the test                
                %disp([gaKA*ga,gbKB*gb]);

                clear('KA','KB');
                
%                ikernel=0;
                if ifeature==1
                  switch ikernel1 
                    case 0
                        TKA=XTest1*XTrain1';  % linear kernel
                    case 1 % polynomial
                        TKA=(XTest1*XTrain1'+1).^idegree1;
                    case 2 % sigmoid
                        TKA=tanh(XTest1*XTrain1'+1);
                    case 3  % Gaussian
                        TKA=XTest1*XTrain1';
                        sigma=xsigma*n1;
                        d1=sum(XTest1.^2,2);
                        d2=sum(XTrain1.^2,2);
                        DKA=d1*ones(1,mtrain)+ones(mtest,1)*d2'-2*TKA;
                        TKA=exp(-DKA/sigma);
                  end
                  switch ikernel2 
                    case 0
                        TKB=XTest2*XTrain2';
                    case 1 % polynomial
                        TKB=(XTest2*XTrain2'+1).^idegree2;
                    case 2 % sigmoid
                        TKB=tanh(XTest2*XTrain2'+1);
                    case 3  % Gaussian
                        sigma=xsigma*n2;
                        TKB=XTest2*XTrain2';
                        d1=sum(XTest2.^2,2);
                        d2=sum(XTrain2.^2,2);
                        DKB=d1*ones(1,mtrain)+ones(mtest,1)*d2'-2*TKB;
                        TKB=exp(-DKB/sigma);
                  end
                else
                  TKA=XTest1;
                  TKB=XTest2;
                end


if isempty(TKA)
                pre2=(gb'*TKB'+(bbm))';

                hit1=0;
                hit2=YTest.*sign(pre2);
                pre=pre2;
                hit=YTest.*sign(pre);
                acorr=(sum(hit)+mtest)/(2*mtest);
                acorr1=0;
                acorr2=(sum(hit2)+mtest)/(2*mtest);
elseif isempty(TKB)
                pre1=(ga'*TKA'+(bam))';

                hit1=YTest.*sign(pre1);
                hit2=0;
                pre=pre1;
                hit=YTest.*sign(pre);
                deci = pre;
                acorr=(sum(hit)+mtest)/(2*mtest);
                acorr1=(sum(hit1)+mtest)/(2*mtest);
                acorr2=0;
else
                pre1=(ga'*TKA'+(bam))';
                pre2=(gb'*TKB'+(bbm))';

                hit1=YTest.*sign(pre1);
                hit2=YTest.*sign(pre2);
                pre=xlambda*pre1+(1-xlambda)*pre2;
                hit=YTest.*sign(pre);
                acorr=(sum(hit)+mtest)/(2*mtest);               
                acorr1=(sum(hit1)+mtest)/(2*mtest);               
                acorr2=(sum(hit2)+mtest)/(2*mtest);               

%                acorr=sum(YTest.*sign(pre1+pre2))/mtest;
end
                disp([acorr,acorr1,acorr2,tacorr,tacorr1,tacorr2]);

% ****************************************************************            
    
function bm=mc_adjust_bias(b,ttpre,gK,YTrain)

                mtrain=max(size(YTrain));
                
                il1err=0;
                ebmax=0.0001;
                xrange=1000;
                ba0=b;
                bsearch1=zeros(xrange,1);
                for iscope=2:5
                    xstep=1/(10^(iscope-1));
                    xbase=xstep*xrange/2;
                
                    for i=1:xrange
                        thit1=YTrain.*sign(ttpre+i*xstep-xbase);
                        tacorr1=(sum(thit1)+mtrain)/(2*mtrain);
                        bsearch1(i)=tacorr1;
                    end
                    bmax=max(bsearch1);
                    bmin=min(bsearch1);
                    ebmax=(bmax-bmin)/1000;
                    fmax1=find(bsearch1>=(bmax-ebmax));
                    imax1=round(median(fmax1));
                    ba0=ba0+imax1*xstep-xbase;
                    ttpre=(gK+ba0)';
                end
                bm=ba0;


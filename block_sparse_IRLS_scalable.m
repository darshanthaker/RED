%BLOCK - SPARSE CODE
function [cs,ca,obj,err_cs,err_ca,ws,wa,Inds,Ds,Da] = block_sparse_IRLS_scalable(x,Ds,Da,classes_no,attacks_no,blck_size,maxIter,lambda1,lambda2,lambda_reg,alg)

%%Block-sparse IRLS algorithm for lp perturbation's classification
%Inputs:  
%    x  : test sample
%    Ds : signal dictionary 
%    Da : attacks  dictionary
%    classes_no  : number of classes
%    attacks_no : number of attack families
%    blck_sizes : size of attacks blocks
%    maxIter : maximum number of iteration
%    lambda1 :  regularization parameter for signal block coefficients 
%    lambda2 : regularization parameter for attacks' blocks coefficients
%    lambda_reg : Tickhonov regularization parameter 
%    alg : selection of algorithm (4 different variants)



%Ouputs:
%   cs : signal coefficients vector 
%   ca :  attacks coefficients vector
%   err_cs : normalized succesive l2 norm difference of cs
%   err_ca :   normalized succesive l2 norm difference of ca
%   ws : diagonal elements of reweighting matrix Ws
%   wa : diagonal elements of reweighting matrix Wa

epsilon = 1e-5; %theshold used for averting division by zero
m = size(Ds,2); %number of columns of Ds
n = size(Da,2); %number of columns of Da

%Random initialization of cs and ca vectors
cs = randn(size(Ds,2),1);
ca = zeros(size(Da,2),1);% 1e-2*randn(size(Da,2),1);
 
err_cs  = zeros(maxIter,1);
err_ca  = zeros(maxIter,1);
blk_1_size = blck_size(1);
converged = 0;
t = 1;
thresh = 1e-5;
epsilon1 = 2; % threshold - negligible energy
Inds = 1:classes_no;
DDA = Da'*Da;
DDS = Ds'*Ds;


while (t <= maxIter) & (converged == 0)
     
     cs_old = cs;
     Ds_old = Ds;
     ca_old = ca;
     Da_old = Da;
     %Selection of algorithm
     if alg==1  %standard block sparse algorithm
         for i=1:classes_no
             norm_cs(i) = norm(cs((m/classes_no)*(i-1)+1: i*(m/classes_no)),2);
             del1 = max(norm_cs(i),epsilon);
             
             
             
             ws((m/classes_no)*(i-1)+1: i*(m/classes_no)) = lambda1/del1; 
            
         for j=1:attacks_no
              norm_ca(i,j) = norm(ca((j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)),2);
              del2 = max(norm_ca(i,j),epsilon);
              wa((j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)) =  lambda1/del2; 
         end
        end
     elseif alg==2 %concatenated cs_i and ca_ijs
         for i=1:classes_no
             ca_ =[];
             for j=1:attacks_no
                 ca_ = [ca_ ca((j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j))];
             end
            norm_csa(i) = norm([cs((m/classes_no)*(i-1)+1: i*(m/classes_no)) ca_],2);
          del1 = max(norm_csa(i),epsilon);
          ws((m/classes_no)*(i-1)+1: i*(m/classes_no)) = lambda1/del1;
          for j=1:attacks_no
          wa((j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)) =  lambda1/del1; 
          end
         end
         
     elseif alg==3 %concatenated cs_i and ca_ijs + regularization of ca_i_j_s
         i=1;
         while i<=classes_no
            ca_ =[];
             for j=1:attacks_no
                 ca_ = lambda2*[ca_ ca((j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j))];
             end
         norm_csa(i) = norm([cs((m/classes_no)*(i-1)+1: i*(m/classes_no)) ca_],2);
         del1(i) = max(norm_csa(i),epsilon);
         
             if del1(i) < epsilon1  
                
                
                 indx_s =  (m/classes_no)*(i-1)+1: i*(m/classes_no);
                 ws(indx_s) = [];
                 Ds(:,indx_s) = [];
                 cs(indx_s) = [];
                 indx_a = [];
                 
                 for j=1:2
                     indx_a = [indx_a  (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)];
                 end
                 wa(indx_a)=[];
                 Da(:,indx_a) =[];
                 ca(indx_a) = [];
                 classes_no = classes_no - 1;
                 m = classes_no*blk_1_size;
                 DDA = Da'*Da;
                 DDS = Ds'*Ds;
                 
             elseif del1(i)>=epsilon1
             
             
             
         ws((m/classes_no)*(i-1)+1: i*(m/classes_no)) = lambda1/del1(i);
         
          for j=1:attacks_no
               norm_ca(i,j) = norm(ca( (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)),2);
               del2 = max(norm_ca(i,j), epsilon);
               wa( (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)) =  (lambda1*lambda2)/del1(i) ;%+ lambda2/del2 ;
          end
          i=i+1;
             end
         end
     elseif alg == 4 %concatenated cs_i and ca_ijs + regularization of ca_i_j_s: second approach
         i=1;
          while i<=classes_no
            for j=1:attacks_no
               
              norm_ca(i,j) = norm(ca(  (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)),2);
            end
             del_s(i) = sqrt(norm(cs((m/classes_no)*(i-1)+1: i*(m/classes_no)))^2 + lambda2*sum(norm_ca(i,:)));
             
             if del_s(i) < epsilon1  
                
                
                 indx_s =  (m/classes_no)*(i-1)+1: i*(m/classes_no);
                 ws(indx_s) = [];
                 Ds(:,indx_s) = [];
                 cs(indx_s) = [];
                 indx_a = [];
                 
                 for j=1:2
                     indx_a = [indx_a  (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)];
                 end
                 wa(indx_a)=[];
                 Da(:,indx_a) =[];
                 ca(indx_a) = [];
                 classes_no = classes_no - 1;
                 m = classes_no*blk_1_size;
                 DDA = Da'*Da;
                 DDS = Ds'*Ds;
                 
             elseif del_s(i)>=epsilon1
             
             ws((m/classes_no)*(i-1)+1: i*(m/classes_no)) = lambda1./del_s(i);
             
            for j=1:attacks_no
              del_1 = sqrt(norm(cs((m/classes_no)*(i-1)+1: i*(m/classes_no)))^2 + sum(norm_ca(i,:)) );
              del_2 = norm_ca(i,j);
              del_a = max(del_1,epsilon);
              del_b = max(del_2,epsilon);
             
              wa(  (j-1)*blk_1_size*classes_no + (i-1)*blck_size(j) + 1 : (j-1)*blk_1_size*classes_no + i*blck_size(j)) =  (lambda2*lambda1)/(del_a*del_b) +lambda_reg ;
            end
            i=i+1;
             end
          end
     end
     
   Ws = diag(ws);
   Wa = diag(wa');
  

   %Updates of ca and cs
   ca = (DDA + Wa )\((Da')*(x - Ds*cs)); 
   cs  = (DDS + Ws)\((Ds')*(x - Da*ca));
   
     
   err_cs(t) =  norm(Ds*cs  - Ds_old*cs_old)/norm(Ds_old*cs_old);
   err_ca(t) = 1;%norm(ca-ca_old)/norm(ca_old);
   if err_cs(t) <= thresh | err_ca(t)<= thresh
       converged = 1;
   end
   obj = 0;
%   obj(t) = .5*norm(x - Da*ca - Ds*cs) + lambda1*sum(norm_cs(:));
   t = t+1;
end

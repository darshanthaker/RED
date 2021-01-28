%BLOCK - SPARSE CODE
function [cs,ca,obj,err_cs,err_ca,ws,wa] = block_sparse_IRLSv2(x,Ds,Da,k,no_attacks,len_a,maxIter,lambda1,lambda2,lambda_reg,flag)
% k : size of csi's
cs = randn(size(Ds,2),1);
ca = zeros(size(Da,2),1);% 1e-2*randn(size(Da,2),1);
m = size(Ds,2);
n = size(Da,2);

len_a_1 = len_a(1);
len_a_2 = len_a(2);
for t=1:maxIter
     
     cs_old = cs;
     ca_old = ca;
    
     
     if flag==1
         for i=1:k
             norm_cs(i) = norm(cs((m/k)*(i-1)+1: i*(m/k)),2);
             del1 = max(norm_cs(i),1e-6);
             ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/del1; 
            
         for j=1:no_attacks
              norm_ca(i,j) = norm(ca((j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)),2);
              del2 = max(norm_ca(i,j),1e-6);
              wa((j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)) =  lambda1/del2; 
         end
        end
     elseif flag==2
         for i=1:k
             ca_ =[];
             for j=1:no_attacks
                 ca_ = [ca_ ca((j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j))];
             end
            norm_csa(i) = norm([cs((m/k)*(i-1)+1: i*(m/k)) ca_],2);
          del1 = max(norm_csa(i),1e-6);
          ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/del1;
          for j=1:no_attacks
          wa((j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)) =  lambda1/del1; 
          end
         end
         
     elseif flag==3
         for i=1:k
            ca_ =[];
             for j=1:no_attacks
                 ca_ = [ca_ ca((j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j))];
             end
         norm_csa(i) = norm([cs((m/k)*(i-1)+1: i*(m/k)) ca_],2);
         del1 = max(norm_csa(i),1e-6);
         ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/del1;
         
          for j=1:no_attacks
               norm_ca(i,j) = norm(ca( (j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)),2);
               del2 = max(norm_ca(i,j), 1e-7);
               wa( (j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)) =  lambda1/del1 + lambda2/del2 ;
          end
         end
     elseif flag == 4
         
          for i=1:k
            for j=1:no_attacks
               
              norm_ca(i,j) = norm(ca(  (j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)),2);
            end
             del_s(i) = max(sqrt(norm(cs((m/k)*(i-1)+1: i*(m/k)))^2 + sum(norm_ca(i,:))),1e-7);
             ws((m/k)*(i-1)+1: i*(m/k)) = lambda1./del_s(i);
            for j=1:no_attacks
              del = max(((norm_ca(i,j))*(sqrt(norm(cs((m/k)*(i-1)+1: i*(m/k)))^2 + sum(norm_ca(i,:)) ))), 1e-7);
              wa(  (j-1)*len_a_1*k + (i-1)*len_a(j) + 1 : (j-1)*len_a_1*k + i*len_a(j)) =  lambda2*lambda1./del +lambda_reg ;
            end
         end
     end
     
   Ws = diag(ws);
   Wa = diag(wa');
  
   ca = (Da'*Da + Wa )\((Da')*(x - Ds*cs)); 
   cs  = (Ds'*Ds + Ws)\((Ds')*(x - Da*ca));
   
     
   err_cs(t) = norm(cs - cs_old)/norm(cs_old);
   err_ca(t) = norm(ca-ca_old)/norm(ca_old);
   obj = 0;
%   obj(t) = .5*norm(x - Da*ca - Ds*cs) + lambda1*sum(norm_cs(:));
   
end

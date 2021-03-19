%BLOCK - SPARSE CODE
function [cs,ca,obj,err_cs,err_ca,ws,wa] = block_sparse_IRLS(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag)
% k : size of csi's
cs = randn(size(Ds,2),1);
ca = randn(size(Da,2),1);
m = size(Ds,2);
n = size(Da,2);
wa = zeros(k,n/k);
for t=1:maxIter
     cs_old = cs;
     ca_old = ca;
     ca = reshape(ca,n/k,k)';
     cs = reshape(cs,m/k,k)';
     
     if flag==1
         for i=1:k
             norm_cs(i) = norm([cs(i,:)],2);
             ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/(norm_cs(i)+eps); 
            
         for j=1:q 
         norm_ca(i,j) = norm(ca(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
         wa(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))) =  lambda1/(norm_ca(i,j) + eps); 
         end
        end
     elseif flag==2
         for i=1:k
            norm_csa(i) = norm([cs(i,:) ca(i,:)],2);
   
          ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/(norm_csa(i)+eps);
          wa(i,1:(n/k)) =  lambda1/(norm_csa(i) + eps); 
         end
         
     elseif flag==3
         for i=1:k
         norm_csa(i) = norm([cs(i,:) ca(i,:)],2);
         ws((m/k)*(i-1)+1: i*(m/k)) = lambda1/(norm_csa(i)+eps);
          for j=1:q
               norm_ca(i,j) = norm(ca(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
               wa(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))) =  lambda1/(norm_csa(i) + eps) + lambda2/(norm_ca(i,j) + eps) ;
          end
         end
         
     end
     
   
   Ws = diag(ws');
   wa = wa';
   Wa = diag(wa(:));
   wa = wa';
   
   ca = ca';
   ca = ca(:);
   cs = cs';
   cs = cs(:);
   cs  = inv(Ds'*Ds + Ws)*(Ds')*(x - Da*ca);
    
   ca = inv(Da'*Da + Wa )*(Da')*(x - Ds*cs);
   
   err_cs(t) = norm(cs - cs_old)/norm(cs_old);
   err_ca(t) = norm(ca-ca_old)/norm(ca_old);
   obj = 0;
%   obj(t) = .5*norm(x - Da*ca - Ds*cs) + lambda1*sum(norm_cs(:));
   
end
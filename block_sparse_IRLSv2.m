%BLOCK - SPARSE CODE
function [cs,ca,obj,err_cs,err_ca,ws,wa] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag)
% k : size of csi's
cs = randn(size(Ds,2),1);
ca = randn(size(Da,2),1);
m = size(Ds,2);
n = size(Da,2);
wa = zeros(k,n/k);
no_attacks = 2;
for t=1:maxIter
     
     cs_old = cs;
     ca_old = ca;
     ca = reshape(ca,n/(no_attacks*k),k*no_attacks)';
     ca = [ca(1:k,:)  ca(k+1:no_attacks*k,:)]; 
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
   wa = [wa(1:n/(no_attacks*k),:) wa(n/(no_attacks*k)+ 1:n/k,:)];
 
   Wa = diag(wa(:));
   wa = wa';
   wa = [wa(1:k,:) wa(k+1:no_attacks*k,:)];
   

   ca = [ca(:,1:(n/(no_attacks*k))); ca(:,n/(no_attacks*k)+1:end)]';
   ca = ca(:);
   cs = cs'; 
   cs = cs(:);
   cs  = (Ds'*Ds + Ws)\((Ds')*(x - Da*ca));
    
   ca = (Da'*Da + Wa )\((Da')*(x - Ds*cs)); 
   
   err_cs(t) = norm(cs - cs_old)/norm(cs_old);
   err_ca(t) = norm(ca-ca_old)/norm(ca_old);
   obj = 0;
%   obj(t) = .5*norm(x - Da*ca - Ds*cs) + lambda1*sum(norm_cs(:));
   
end
%test block sparse red
clear all; close all;clc;
randn('seed',1);
d= 500;
m = 40;
n = 250;
k = 5;%size of cs_is is (m/k x 1)
q = 5;%size of ca_{ij}s is (n/(k*q) x 1)

Ds = randn(d,m);
Da = randn(d,n);


cs_1 = 1e0*randn(m/k,1);
cs = [cs_1 ; zeros(m-m/k,1)];



ca_1 = 1e-4*randn(n/(k*q),1);
ca_3 = 1e-4*randn(n/(k*q),1);
ca = [ ca_1 ; zeros(n/(k*q),1); ca_3; zeros(n-3*n/(k*q),1)];
Ds = normc(Ds);
Da = normc(Da);

noise  = 6e-5*randn(d,1);
signal = Ds*cs + Da*ca ;
x = signal +  noise;
SNR = snr(signal -Ds*cs,noise)

%------------------------
flag =1;
maxIter = 500;
lambda1 = 0.00014;% 0.01;
lambda2 = 0;


[cs_est_1,ca_est_1,obj,err_cs_1,err_ca_1,ws_1,wa_1] = block_sparse_IRLS(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);
%------------------------
flag =2;
maxIter = 500;
lambda1 = 0.001;% 0.01;
lambda2 = 0;

[cs_est_2,ca_est_2,obj,err_cs_2,err_ca_2,ws_2,wa_2] = block_sparse_IRLS(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);

%-------------------
flag =3;
maxIter = 500;
lambda1 = 0.00026;% 
lambda2 = 0.00016;

[cs_est_3,ca_est_3,obj,err_cs_3,err_ca_3,ws_3,wa_3] = block_sparse_IRLS(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);

%------------------
maxIter = 500;
lambda1 = 0;% 0.01;
lambda2 = 0;
[cs_est_no_reg,ca_est_no_reg,obj,err_cs,err_ca,ws,wa] = block_sparse_IRLS(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);



%obj_true = lambda1*sum(norm_cs(:));


%-------
figure(1);subplot(321);imagesc(ca);colorbar;title('true ca');subplot(322);imagesc(ca_est_1);colorbar;caption = sprintf('Estimated ca -1, error = %.3f', norm(ca-ca_est_1)/norm(ca));title(caption);
subplot(323);imagesc(ca_est_2);colorbar;caption = sprintf('Estimated ca -2, error = %.3f', norm(ca-ca_est_2)/norm(ca));title(caption);subplot(324);imagesc(ca_est_3);colorbar;caption = sprintf('Estimated ca -3, error = %.3f', norm(ca-ca_est_3)/norm(ca));title(caption);
subplot(325);imagesc(ca_est_no_reg);colorbar;caption = sprintf('Estimated ca - No-Reg, error = %.3f', norm(ca-ca_est_no_reg)/norm(ca));title(caption);
figure(2);subplot(321); imagesc(cs);colorbar;title('true cs');subplot(322);imagesc(cs_est_1);colorbar;caption = sprintf('Estimated cs -1, error = %d', norm(cs-cs_est_1)/norm(cs));title(caption);
subplot(323);imagesc(cs_est_2);colorbar;caption = sprintf('Estimated cs -2, error = %d', norm(cs-cs_est_2)/norm(cs));title(caption);subplot(324);imagesc(cs_est_3);colorbar;caption = sprintf('Estimated cs -3, error = %d', norm(cs-cs_est_3)/norm(cs));title(caption);subplot(325);imagesc(cs_est_no_reg);
colorbar;caption = sprintf('Estimated cs -No-reg, error = %d', norm(cs-cs_est_no_reg)/norm(cs));title(caption);


ca_est_1 = reshape(ca_est_1,n/k,k)';
ca_est_2 = reshape(ca_est_2,n/k,k)';
ca_est_3 = reshape(ca_est_3,n/k,k)';
ca_est_no_reg= reshape(ca_est_no_reg,n/k,k)';
ca_true = reshape(ca,n/k,k)';

norm_ca_1 = zeros(k,q);
norm_ca_2 = zeros(k,q);
norm_ca_3 = zeros(k,q);
norm_ca_4 = zeros(k,q);
norm_ca_true = zeros(k,q);
 for i=1:k
     for j=1:q
       norm_ca_1(i,j) = norm(ca_est_1(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_2(i,j) = norm(ca_est_2(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_3(i,j) = norm(ca_est_3(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_4(i,j) = norm(ca_est_no_reg(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_true(i,j) = norm(ca_true(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
    end
     
 end
 bl = k*q;
  norm_ca_1 = norm_ca_1';norm_ca_2 = norm_ca_2';norm_ca_3 = norm_ca_3';norm_ca_4 = norm_ca_4';norm_ca_true = norm_ca_true';
  
  figure(3);  stem([norm_ca_1(:),norm_ca_2(:),norm_ca_3(:),norm_ca_4(:),norm_ca_true(:)],'filled')
  legend('energy 1','energy 2','energy 3','energy no reg','True');
       
%test block sparse red
%clear all; close all;clc;
%close all;clc;
tic
randn('seed',1);warning off;
d= 1400;
m = 2090;
n = 4180;
k = 38;%size of cs_is is (m/k x 1)
q = 2;%size of ca_{ij}s is (n/(k*q) x 1)
no_attacks=2;
%load('home/paris/singal_dict.mat');
%load('home/paris/attack_dict.mat');
%Ds = signal_dict;
%Da = attack_dict;
%Ds = randn(d,m);
%Da = randn(d,n);


cs_1 = 1e0*randn(m/k,1);
cs = [ zeros(m-m/k,1) ;cs_1 ];

ca_1 = 1e-4*randn(n/(k*q),1);
ca_3 = 1e-4*randn(n/(k*q),1);
 ca = [ zeros(n/(no_attacks),1); zeros(n/no_attacks - n/(k*q),1); ca_1 ; ];

%Ds = normc(Ds);
%Da = normc(Da);

noise  = 3e-5*randn(d,1);
signal = Ds*cs + Da*ca ;
x = signal +  noise;
SNR = snr(signal -Ds*cs,noise)

%------------------------
flag =1
maxIter = 100;
lambda1 = 0.11%0.00014;% 0.01;
lambda2 = 0;


[cs_est_1,ca_est_1,obj,err_cs_1,err_ca_1,ws_1,wa_1] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);
%------------------------
flag =2
maxIter = 100;
lambda1 = 0.11;%0.001;% 0.01;
lambda2 = 0;

[cs_est_2,ca_est_2,obj,err_cs_2,err_ca_2,ws_2,wa_2] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);

%-------------------
flag =3
maxIter = 100;
lambda1 = 0.11%0.00026;% 
lambda2 = 0.2%0.01e0%0.00016;

[cs_est_3,ca_est_3,obj,err_cs_3,err_ca_3,ws_3,wa_3] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);

flag =4
maxIter = 100;
lambda1 = 0.5%0.00026;% 
lambda2 = 0.0%0.01e0%0.00016;

[cs_est_4,ca_est_4,obj,err_cs_4,err_ca_4,ws_4,wa_4] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);

%------------------
disp('no reg')
maxIter = 100;
lambda1 = 0;% 0.01;
lambda2 = 0;
%[cs_est_no_reg,ca_est_no_reg,obj,err_cs,err_ca,ws,wa] = block_sparse_IRLSv2(x,Ds,Da,k,q,maxIter,lambda1,lambda2,flag);



%obj_true = lambda1*sum(norm_cs(:));


%-------
figure(1);subplot(321);imagesc(ca);colorbar;title('true ca');subplot(322);imagesc(ca_est_1);colorbar;caption = sprintf('Estimated ca -1, error = %.3f', norm(ca-ca_est_1)/norm(ca));title(caption);
subplot(323);imagesc(ca_est_2);colorbar;caption = sprintf('Estimated ca -2, error = %.3f', norm(ca-ca_est_2)/norm(ca));title(caption);subplot(324);imagesc(ca_est_3);colorbar;caption = sprintf('Estimated ca -3, error = %.3f', norm(ca-ca_est_3)/norm(ca));title(caption);
subplot(325);imagesc(ca_est_4);colorbar;caption = sprintf('Estimated ca - 4, error = %.3f', norm(ca-ca_est_4)/norm(ca));title(caption);
figure(2);subplot(321); imagesc(cs);colorbar;title('true cs');subplot(322);imagesc(cs_est_1);colorbar;caption = sprintf('Estimated cs -1, error = %d', norm(cs-cs_est_1)/norm(cs));title(caption);
subplot(323);imagesc(cs_est_2);colorbar;caption = sprintf('Estimated cs -2, error = %d', norm(cs-cs_est_2)/norm(cs));title(caption);subplot(324);imagesc(cs_est_3);colorbar;caption = sprintf('Estimated cs -3, error = %d', norm(cs-cs_est_3)/norm(cs));title(caption);subplot(325);imagesc(cs_est_4);
colorbar;caption = sprintf('Estimated cs -4, error = %d', norm(cs-cs_est_4)/norm(cs));title(caption);

ca_est_1 = reshape(ca_est_1,n/(no_attacks*k),k*no_attacks)';
ca_est_1 = [ca_est_1(1:k,:)  ca_est_1(k+1:no_attacks*k,:)];

ca_est_2 = reshape(ca_est_2,n/(no_attacks*k),k*no_attacks)';
ca_est_2 = [ca_est_2(1:k,:)  ca_est_2(k+1:no_attacks*k,:)];

ca_est_3 = reshape(ca_est_3,n/(no_attacks*k),k*no_attacks)';
ca_est_3 = [ca_est_3(1:k,:)  ca_est_3(k+1:no_attacks*k,:)];

ca_est_no_reg = reshape(ca_est_no_reg,n/(no_attacks*k),k*no_attacks)';
ca_est_no_reg = [ca_est_no_reg(1:k,:)  ca_est_no_reg(k+1:no_attacks*k,:)];

ca_est_4 = reshape(ca_est_4,n/(no_attacks*k),k*no_attacks)';
ca_est_4 = [ca_est_4(1:k,:)  ca_est_4(k+1:no_attacks*k,:)];


ca = reshape(ca,n/(no_attacks*k),k*no_attacks)';
ca = [ca(1:k,:)  ca(k+1:no_attacks*k,:)];
cs = reshape(cs,m/k,k)';
cs_est_3 = reshape(cs_est_3,m/k,k)';
cs_est_1 = reshape(cs_est_1,m/k,k)';
cs_est_2 = reshape(cs_est_2,m/k,k)';
cs_est_4 = reshape(cs_est_4,m/k,k)';
cs_est_ = reshape(cs_est_3,m/k,k)';
     
norm_ca_1 = zeros(k,q);
norm_ca_2 = zeros(k,q);
norm_ca_3 = zeros(k,q);
norm_ca_4 = zeros(k,q);
norm_ca_true = zeros(k,q);
 for i=1:k
      norm_cs_1(i) = norm(cs_est_1(i,:),2);
      norm_cs_2(i) = norm(cs_est_2(i,:),2);
      norm_cs_3(i) = norm(cs_est_3(i,:),2);
      norm_cs_4(i) = norm(cs_est_4(i,:),2);
      norm_cs_true(i) = norm(cs(i,:),2);
     for j=1:q
       norm_ca_1(i,j) = norm(ca_est_1(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_2(i,j) = norm(ca_est_2(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_3(i,j) = norm(ca_est_3(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
      norm_ca_4(i,j) = norm(ca_est_4(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
       norm_ca_true(i,j) = norm(ca(i,(n/(q*k))*(j-1)+1: j*(n/(k*q))),2);
    end
     
 end
 bl = k*q;
  figure(3);  stem([norm_ca_1(:),norm_ca_2(:),norm_ca_3(:),norm_ca_4(:),norm_ca_true(:)],'filled')
  legend('energy 1','energy 2','energy 3','energy 4','True');
  figure(4); stem([norm_cs_1(:),norm_cs_2(:),norm_cs_3(:),norm_cs_4(:),norm_cs_true(:)],'filled');
legend('energy 1','energy 2','energy 3','energy 4','True');
       
wa_3 = wa_3';
figure(5);subplot(121);plot(1:n,wa_3(:));axis tight;subplot(122);plot(1:m,ws_3);axis tight;

wa_4 = wa_4';
figure(6);subplot(121);plot(1:n,wa_4(:));axis tight;subplot(122);plot(1:m,ws_4);axis tight;

toc

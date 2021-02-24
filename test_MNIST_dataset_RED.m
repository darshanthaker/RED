
%% Classification of lp attacks on MNIST dataset
warning off;
clear all;
rand('seed',10);
flag =3; %select algorithm
maxIter = 300; %maximum number of iterations
lambda1 =4.5;%regularization parameter used in c_s and c_a updates
lambda2 =3  ; %1 %regularization parameter for c_a updates
no_attacks=2; %number of different attacks
classes_no=10; %number of different classes
lambda_reg = 0.01;%0.01 %tickhonov regularization parameter used in algorithm 4 (flag==4)

load('Users/paris/Google Drive/Hyppocrates/Ongoing/RED/Ds_sub200mnist.mat'); %load signal dictionary
Ds =  data;%signal dictionary
clear data;


load('Users/paris/Google Drive/Hyppocrates/Ongoing/RED/Da_sub200_2infmnist.mat'); %load attack dictionary
Da =  data;  %attack's dictionary
clear data;
  
m = size(Ds,2);%number of columns of signal dictionary
n = size(Da,2);%number of columns of attack's dictionary


%load test sets
%load l2 pertubed test samples
load('Users/paris/Google Drive/Hyppocrates/Ongoing/RED/l2_eps0.3mnist.mat');
l2_attacked_ims = squeeze(data);
clear data;

%load linf pertubed test samples
load('Users/paris/Google Drive/Hyppocrates/Ongoing/RED/linf_eps0.3mnist.mat');
linf_attacked_ims = squeeze(data);
clear data;

%load linf pertubed test samples
load('Users/paris/Google Drive/Hyppocrates/Ongoing/RED/test_ymnist.mat');
test_y = data;
clear data;


siz_blk_s = 200; %size of signal blocks
%%
%samp = randperm(324,100);
tic
for t = 1:100%1:100%:1 %iterate over test samples 
     t
     x = linf_attacked_ims(t,:,:); %linf_attacked_ims(t,:,:)
     x = squeeze(x);
     x = x';
     x = x(:);

att_blck_size = [200,200]; %sizes of blocks of different attacks families    
     tic 
[cs_est_4,ca_est_4,obj,err_cs_4,err_ca_4,ws_4,wa_4,Inds,Ds_e,Da_e] = block_sparse_IRLS_scalable(x,Ds,Da,classes_no,no_attacks,att_blck_size,maxIter,lambda1,lambda2,lambda_reg,flag);
toc


m = size(Ds_e,2);
k = m/siz_blk_s;
Indx_s =[];
for i=1:k
%Ds_e_  = repmat(Ds_e(:,(m/k)*(i-1)+1: i*(m/k)),1,classes_no);

[~,inds] = ismember(Ds_e(10,(m/k)*(i-1)+1: i*(m/k)),Ds(10,:));
Indx_s = [Indx_s  ((inds(1)-1)/siz_blk_s)];
end


xrec = Ds_e*cs_est_4 + Da_e*ca_est_4;%reconstruted x

%%

 cs = cs_est_4;
 ca = ca_est_4;
 
blk_1_size = att_blck_size(1); %size of 1st threat model
norm_ca_4 = zeros(k,no_attacks);
norm_cs_4 = zeros(k,1);

 for i=1:k
      ind_sig_i = (m/k)*(i-1)+1: i*(m/k);%indices of i-th signal block
     
      norm_cs_4(i) = norm(cs(ind_sig_i),2); %energy of i-th signal block
     
     for j=1:no_attacks
         
         ind_att_ij = (j-1)*blk_1_size*k + (i-1)*att_blck_size(j) + 1 : (j-1)*blk_1_size*k + i*att_blck_size(j);%indices of i-th,j-th attack block
        
         norm_ca_4(i,j) = norm(ca(ind_att_ij),2);  %energy of i-th,j-th attack block
    end
     
 end
 %Plot energies of blocks of signal and attack blocks
% figure(1);  stem(norm_ca_4(:),'filled');axis tight
%  legend('l2 norm of attack blocks');
 % 
 % figure(2); stem(Indx_s,norm_cs_4(:),'filled','r');axis tight;
 %legend('l2 norm of signal blocks');
 
%%
%Signal and attack classifiers based on reconstruction errors
err_class = zeros(k,1);
for i=1:k
    ind_sig_i = (m/k)*(i-1)+1: i*(m/k);%indices of i-th signal block
    err_class(i) = norm(x - Ds_e(:, ind_sig_i)*cs(ind_sig_i)-Da_e*ca);

end

[mini ith(t)] = min(err_class);%signal class for for t-th test sample is ith(t)

class_pred(t) = Indx_s(ith(t))
test_y(t)
err_attack = zeros(2,1);
for j=1:2
    ind_att_ij_ = (j-1)*blk_1_size*k + (ith(t)-1)*att_blck_size(j) + 1 : (j-1)*blk_1_size*k + ith(t)*att_blck_size(j);
err_attack(j)  = norm(x - Ds_e*cs - Da_e(:,ind_att_ij_)*ca(ind_att_ij_)); 
end
[mina att_pred(t)] =min(err_attack);%attacks class for for t-th test sample is jth(t)
%ith
%jth

end
toc
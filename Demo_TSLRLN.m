% clc;
clear;
addpath(genpath('TSLRLN'));
addpath(genpath('Dataset'));
addpath(genpath('BM3D'));
addpath(genpath('BM4D'));
addpath('Quantitative Assessment');
load('WDC_ORI.mat');

[nr,nc,L] = size(Img);
Img = Img./repmat(max(max(Img,[],1),[],2),nr,nc);
% Simulation case 1: the variances of Gaussian noise were randomly selected ranging from 0.15 to 0.25.
sig =  0.15+0.1*rand(L,1);
for i = 1 : L
    Noisy_Img(:,:,i) = Img(:,:,i) + sig(i)*randn(nr,nc);
end
sigma = mean(sig);
[mpsnr,psnr] = MPSNR(Img,Noisy_Img);
[mssim,ssim] = MSSIM(Img,Noisy_Img);
% [mfsim,fsim] = MFSIM(Img,Noisy_Img);
% ergas = ErrRelGlobAdimSyn(Img,Noisy_Img);
% msa = MSA(Img, Noisy_Img);
disp('WDC Case 1:');
disp(['Noisy: MPSNR = ' num2str(mpsnr) '; MSSIM = ' num2str(mssim) '; Mean Variance = ' num2str(sigma)]);

X = Img;
Y = Noisy_Img;
clear Img Noisy_Img;

% parameters should be tuned according to the data itself to reach optimal results.
lambda=10e5;
gamma=0.01;
sub_dim=6;
iter=3;

tic
Ys_TSLRLN = TSLRLN_fast(Y, lambda, gamma, sub_dim, iter);
time= toc;

[mpsnr1,psnr1] = MPSNR(X,Ys_TSLRLN);
[mssim1,ssim1] = MSSIM(X,Ys_TSLRLN);
% [mfsim1,fsim1] = MFSIM(X,Ys_TSLRLN);
% ergas1 = ErrRelGlobAdimSyn(X,Ys_TSLRLN);
% msa1 = MSA(X, Ys_TSLRLN);
disp(['Denoised: MPSNR = ' num2str(mpsnr1) '; MSSIM = ' num2str(mssim1) '; TIME = ' num2str(time) newline]);
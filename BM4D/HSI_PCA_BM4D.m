function Ys = HSI_PCA_BM4D(Y,k_num)
% Y is the 3D HSI data nr * nc *L
[nr, nc, L]= size(Y);
[PC, PAR] = PCA_img(Y,'all');
PC_BM4D = PC(:,:,k_num+1:end);
% using BM4D to denoise the rest PCs by estimating the noise
PC_est = bm4d(PC_BM4D, 'Gauss', 0);
PC(:,:,k_num+1:end) = PC_est;
% inverse PCA
Ys = reshape(PC,nr*nc,L)*PAR.eigvec'+repmat(PAR.meanvalues,1,nr*nc)';
Ys = reshape(Ys,nr,nc,L);
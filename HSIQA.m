function [psnr, ssim,mfsim,sam,ergas] = HSIQA(imagery1, imagery2)

%==========================================================================
% Evaluates the quality assessment indices for two HSIs.
% Input:
%   imagery1 - the reference HSI data array
%   imagery2 - the target HSI data array
% NOTE: HSI data array  is a M*N*K array for imagery with M*N spatial
%	pixels, K bands and DYNAMIC RANGE [0, 255]. If imagery1 and imagery2
%	have different size, the larger one will be truncated to fit the
%	smaller one.
%
% Output:
%   psnr - Peak Signal-to-Noise Ratio
%   ssim - Structure SIMilarity
%   msam - Spectral Angle Mapper
%==========================================================================

[m, n, k] = size(imagery1);
[mm, nn, kk] = size(imagery2);
m = min(m, mm);
n = min(n, nn);
k = min(k, kk);
imagery1 = imagery1(1:m, 1:n, 1:k);
imagery2 = imagery2(1:m, 1:n, 1:k);

psnr = 0;
ssim = 0;fsim=0;
for i = 1:k
    psnr = psnr + 10*log10(255^2/mse(imagery1(:, :, i) - imagery2(:, :, i)));
    ssim = ssim + ssim_index(imagery1(:, :, i), imagery2(:, :, i));
    fsim=fsim+FeatureSIM(imagery1(:, :, i), imagery2(:, :, i));
end
psnr = psnr/k;
ssim = ssim/k;
mfsim=fsim/k;
sam = SAM3D(imagery1/255, imagery2/255);
ergas = ErrRelGlobAdimSyn(imagery1, imagery2);
end

clc,clear
addpath('quality_assess');

%case_num=6;
%load(strcat('case',num2str(case_num),'.mat'));

load Noisy_WDC_CASE6
%load Noisy_Pavia_CASE6

[psnr1,ssim1,sam1,ergas1,mfsim1] = HSIQA(img*255, Nhsi_6*255);
fprintf('quality_img1: psnr= %f ssim=%f sam=%f ergas=%f mfsim=%f \n',psnr1,ssim1,sam1,ergas1,mfsim1);
%psnr_noisy=MPSNR(img,Nhsi_6);
%fprintf('quality_img1: psnr= %f \n',psnr_noisy);
[n1,n2,n3]=size(img);

opts=[];
opts.tau=300/sqrt(n1*n2); %S [10 1000] choose 300
opts.alpha=30/sqrt(n1*n2); %spatial group [10 100] choose 30
opts.beta=1.5; % spectral
opts.mu=0.01;
opts.lambda=5; % X [ ]
opts.tol=1e-4;
opts.R=3;
%opts.lambda2=0.5;
opts.img=img;

out_image=BTD_EGS(Nhsi_6,opts);
%[psnr2,ssim2,sam2,ergas2,mfsim2] = HSIQA(img*255,out_image*255);
%fprintf('quality_img2: psnr= %f ssim=%f sam=%f ergas=%f mfsim=%f \n',psnr2,ssim2,sam2,ergas2,mfsim2);
psnr_denoising=MPSNR(img,out_image);
fprintf('quality_img2: psnr= %f\n',psnr_denoising);
Re_msi{1}=Nhsi_6; Re_msi{2}=out_image; methodname={'noisy','BTD_EGS'};
enList=[1,2];
figure,
showMSIResult(Re_msi,methodname,enList,45,n3);


[psnr, ssim,mfsim,sam,ergas] = HSIQA(img*255,out_image*255);


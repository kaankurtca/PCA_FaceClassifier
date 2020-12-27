clc; clear all; close all;
 
nasacolor=imread('TarantulaNebula.jpg');
figure; image(nasacolor);
 
nasa=sum(nasacolor,3,'double');
m=max(max(nasa));
nasa=nasa*255/m; %rgb to gray scale
 
figure; image(nasa); colormap(gray(256));
title('Grayscale NASA photo');
 
[U,S,V]=svd(nasa); %we apply svd function to get eigenfaces(eigenvectors) and eigen values
figure; semilogy(diag(S)); %we plot eigenvalues as semilogarithmic to observe rapidly dropping off
 
nasa100=U(:,1:100)*S(1:100,1:100)*V(:,1:100)';
nasa50=U(:,1:50)*S(1:50,1:50)*V(:,1:50)';
nasa25=U(:,1:25)*S(1:25,1:25)*V(:,1:25)';
% we get nasa images that include 25, 50 and 100 eigenfaces. 
 
figure; image(nasa25); colormap(gray(256));
title('25 eigenfaces');
 
figure; image(nasa50); colormap(gray(256));
title('50 eigenfaces');
 
figure; image(nasa100); colormap(gray(256));
title('100 eigenfaces');

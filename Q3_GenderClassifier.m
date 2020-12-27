clc;clear all;close all;

menFolder = 'faces_men'; listname1 = dir(fullfile(menFolder,'*.jpg'));
men=[];
for k = 1:length(listname1)
    man=reshape(imread([menFolder filesep listname1(k).name]),[1,1296]);
    men=[men; man];
end

womenFolder = 'faces_women'; listname2 = dir(fullfile(womenFolder,'*.jpg'));
women=[];
for k = 1:length(listname2)
    woman=reshape(imread([womenFolder filesep listname2(k).name]),[1,1296]);
    women=[women; woman];
end
faces=double([men; women]);
%We read image files as a vector and store these in faces matrix
faceW = 36;  faceH = 36;
numFaces=400; numPerLine = 10;  ShowLine = 2; 

Y = zeros(faceH*ShowLine,faceW*numPerLine);
Z = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(men(i*numPerLine+j+1,:),[faceH,faceW]);
        Z(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(women(i*numPerLine+j+1,:),[faceH,faceW]); 
  	end 
end 
subplot(2,1,1); imagesc(Y);colormap(gray);
subplot(2,1,2);imagesc(Z);colormap(gray); %Here, we plot images in one figure.

men = [1:200];  women = [200:400];
% Subtract the mean 'face' before performing PCA
h = 36; w = 36;
meanFace = mean(faces, 1);
faces = faces - repmat(meanFace, numFaces, 1);

[u,d,v] = svd(faces.', 'econ'); %we give the 'econ' as the second parameter to get 'u' matrix that has 1024 rows
eigVals = diag(d);
eigVecs = u; % Pull out eigen values and vectors

% Plot the mean sample and the first three principal components
figure; imagesc(reshape(meanFace, h, w)); colormap(gray); title('Mean Face');
figure;
subplot(1, 3, 1); imagesc(reshape(u(:, 1), h, w)); colormap(gray);title('First Eigenface');
subplot(1, 3, 2); imagesc(reshape(u(:, 2), h, w)); colormap(gray);title('Second Eigenface');
subplot(1, 3, 3); imagesc(reshape(u(:, 3), h, w)); colormap(gray);title('Third Eigenface');

womenFaces = faces(women, :); menFaces = faces(men, :);
womenWeights = eigVecs(:,200:400) * womenFaces;
menWeights = eigVecs(:,1:200) * menFaces;

for i = 1:length(men)
    test_men=menWeights(:,i);
    test_repeat_men=repmat(test_men,1,(length(men)-1));
    men_weights_no_test=[menWeights(:,1:i-1) menWeights(:,i+1:end)];
    distance_men=test_repeat_men-men_weights_no_test(:,length(men)-1);
    distance_men_val(i)=sum(vecnorm(distance_men))/(length(men)-1);
end

for i = 1:length(men)
    test_men=menWeights(:,i);
    test_repeat_women=repmat(test_men,1,(length(women)));
    distance_women=test_repeat_women-womenWeights(:,1:length(men)+1);
    distance_women_val(i)=sum(vecnorm(distance_women))/(length(women));
end

for i = 1:length(men)
   decision(i)=distance_women_val(i)>= distance_men_val(i); %if this  condition is true, desicion(i) will equal to 1 and label men class
end
accuracy=sum(decision)/length(men); % it gives a score that tells us how accurately we can predict the smiling class.

decision
fprintf("Accuracy score for men faces: %.2f\n",accuracy);
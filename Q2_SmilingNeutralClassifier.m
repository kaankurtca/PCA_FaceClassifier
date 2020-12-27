clc; clear all; close all;

s = load('Yale.mat','fea','gnd');
face=s.fea; label=s.gnd;
s_ind=3:11:157; n_ind=6:11:160;
sn_ind=[s_ind, n_ind]; faces=face(sn_ind,:); %Here, we load the matrix that contains every images as a vector, and we extract the smiling and neutral images.

faceW = 32;  faceH = 32;
numFaces=30; numPerLine = 11;  ShowLine = 2; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(faces(i*numPerLine+j+1,:),[faceH,faceW]); 
  	end 
end 
imagesc(Y);colormap(gray); %Here, we plot images in one figure.

neutral = [16:30];
smile = [1:15];

% Subtract the mean 'face' before performing PCA
h = 32; w = 32;
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

neutralFaces = faces(neutral, :); smileFaces = faces(smile, :);
neutralWeights = eigVecs(:,16:30) * neutralFaces;
smileWeights = eigVecs(:,1:15) * smileFaces;

for i = 1:length(smile)
test_smile=smileWeights(:,i);
test_repeat_smile=repmat(test_smile,1,(length(smile)-1));
smile_weights_no_test=[smileWeights(:,1:i-1) smileWeights(:,i+1:end)];
distance_smile=test_repeat_smile-smile_weights_no_test(:,1:14);
distance_smile_val(i)=sum(vecnorm(distance_smile))/(length(smile)-1);
end

for i = 1:length(smile)
test_smile=smileWeights(:,i);
test_repeat_neutral=repmat(test_smile,1,(length(neutral)));
distance_neutral=test_repeat_neutral-neutralWeights(:,1:15);
distance_neutral_val(i)=sum(vecnorm(distance_neutral))/(length(neutral));
end

for i = 1:length(smile)
   decision(i)=distance_neutral_val(i)>= distance_smile_val(i); %if this  condition is true, desicion(i) will equal to 1 and label smiling class
end
accuracy=sum(decision)/length(smile); % it gives a score that tells us how accurately we can predict the smiling class.

decision
fprintf("Accuracy score for smiling faces: %.2f\n",accuracy);
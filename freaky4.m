% I = imread('th.jpeg'); 
% corners = detectFASTFeatures(I); 
% [features, valid_points] = extractFeatures(I, corners);
% % [corners, scores] = fast9(I, 30,1);
% % [features, valid_points] = extractFeatures(I, corners, 'Method', 'FREAK');
%  figure; imshow(I); hold on;
%     plot(valid_points.selectStrongest(50));

%  I = imread('trex.png');
%  J = imread('trex2.png');
%  
%  grayImage = rgb2gray(I);
%  grayImage2 = rgb2gray(J);
%  
%  points = detectBRISKFeatures(grayImage);
%  points2 = detectBRISKFeatures(grayImage2);
%  
%  [features, valid_points] = extractFeatures(grayImage, points, 'Method', 'FREAK');
%  [features2, valid_points2] = extractFeatures(grayImage2, points2, 'Method', 'FREAK');
%  
%  indexPairs = matchFeatures(features, features2);
%  
%  matchedPoints = valid_points(indexPairs(:, 1), :);
%  matchedPoints2 = valid_points2(indexPairs(:, 2), :);
%  
%  figure;
%  set(gcf, 'Position', get(0,'Screensize')); 
%  
%  subplot(1,3,1);
%  imshow(I); title('First Image');
%  
%  subplot(1,3,2);
%  imshow(J); title('Second Image');
%  
%  subplot(1,3,3);
%  showMatchedFeatures(grayImage, grayImage2, matchedPoints, matchedPoints2);
%  title('FAST detector and BRISK descriptor');
 
 


%-----------------------------
%  set(gcf, 'Position', get(0,'Screensize')); 
%  
%  subplot(1,3,1);
%  imshow(I); title('Original Image');
%   
%  subplot(1,3,2);
%  imshow(grayImage); title('BRISK detector'); hold on;
%  plot(points.selectStrongest(20));
%  
%  subplot(1,3,3); 
%  imshow(grayImage); title('FREAK descriptor'); hold on;
%  plot(valid_points.selectStrongest(20));
%  -----------------------------


% Load images.
% buildingDir = fullfile('C','Users','Sonam','Desktop','pictures2');
%buildingScene = imageSet(buildingDir);

buildingScene{1}=imread('pictures2\32.jpg');
buildingScene{2}=imread('pictures2\31.jpg');
% buildingScene{3}=imread('pictures2\12.jpg');
% buildingScene{3}=imread('pictures\flower_3.jpg');

% buildingScene={image1,image2,image3};

% Display images to be stitched
% montage(buildingScene.ImageLocation)

% Read the first image from the image set.
I = buildingScene{1};

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
tforms(length(buildingScene)) = projective2d(eye(3));

% Iterate over remaining image pairs
for n = 2:length(buildingScene)

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = buildingScene{n};

    % Detect and extract SURF features for I(n).
    grayImage = rgb2gray(I);
    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
%     indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    indexPairs = matchFeatures(features, featuresPrevious);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

    % Compute T(1) * ... * T(n-1) * T(n)
    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

imageSize = size(I);  % all the images are the same size

% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end


for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);


blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:length(buildingScene)

    I = buildingScene{i};

    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Create an mask for the overlay operation.
    warpedMask = imwarp(ones(size(I(:,:,1))), tforms(i), 'OutputView', panoramaView);

    % Clean up edge artifacts in the mask and convert to a binary image.
    warpedMask = warpedMask >= 1;

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, warpedMask);
end

figure
imshow(panorama)
% disp(length(buildingScene));
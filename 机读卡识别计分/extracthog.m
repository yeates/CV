function Features = extracthog(imgSet,numImages,imgsize,cellSize,hogFeatureSize)
    Features = zeros(numImages, hogFeatureSize, 'single');
    for i = 1:numImages
        img = readImage(imgSet, i);
        %图像应该是一样的大小，以保证hog维数一样
        if sum(abs(size(img)-imgsize)) ~= 0
            img = imresize(img,imgsize);%图像大小归一化
        end
        % Apply pre-processing steps
       % img = im2bw(img,graythresh(img));
        Features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
      %  subplot(2,5,i);imshow(img)
    end
end
function Features = extracthog_frame(imgFrame, Bg, imgsize, cellSize, hogFeatureSize)
    Features = zeros(size(imgFrame, 1), hogFeatureSize, 'single');
    for i = 1:size(imgFrame, 1)
        img = imcrop(Bg, imgFrame(i, :));
        %图像应该是一样的大小，以保证hog维数一样
        if sum(abs(size(img)-imgsize)) ~= 0
            img = imresize(img,imgsize);%图像大小归一化
        end
        Features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
end


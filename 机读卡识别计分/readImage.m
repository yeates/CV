function img = readImage(imgSet, i)
    img = imread(imgSet(i).File);
    if size(img,3)>1
       img = rgb2gray(img);
    end
end
close all;
clc;

FileNames_1 = dir('./img/');
FileNames_2 = dir('./img/测试图片');
FileNames_3 = dir('./img/新模板测试图片');

%WindowXY = floor(sqrt(length(FileNames_2)));
%figure();
for n=3:length(FileNames_2)
    file = ['./img/测试图片/',FileNames_2(n).name];
    I = imread(file);
    I = imresize(I, [640, 360]);    % 图像大小修改，防止分辨率过大使运行时间太长
    [height, width, pass] = size(I);
    if pass > 1 % 判断是否是灰度图像
        I = rgb2gray(I); % 灰度转换 
    end
    Level = graythresh(I);
    Bg = adaptthresh(I, Level, 'Statistic', 'median',  'ForegroundPolarity','dark'); % 自适应滤波
    Bg = Bg * 255;
    Bg = uint8(Bg);
    tempI = 255 - Bg + I;
    Binary = im2double(tempI).^16;   % 拉伸
    %WindowPos = mod(n, WindowXY) + floor(n / WindowXY);
    figure();
    %subplot(WindowXY + 1, WindowXY + 1, WindowPos); 
    imshow(Binary); title(FileNames_2(n).name);
end
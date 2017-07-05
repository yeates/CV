
close all;
clc;

I = imread('./img/测试图片/魅蓝-P2 (3).jpg');
I = imresize(I, [640, 360]);    % 图像大小修改，防止分辨率过大使运行时间太长
[M, N, pass] = size(I);
if pass > 1 % 判断是否是灰度图像
    I = rgb2gray(I); % 灰度转换 
end
%I = histeq(I); % 图像增强，消除手机照相能力的影响
figure();
subplot(131);
imshow(I);title('原图');
Level = graythresh(I);
AdaptT = adaptthresh(I, 'Statistic', 'median',  'ForegroundPolarity', 'dark'); % 自适应阈值
Bg = AdaptT * 255; % 扩展
Bg = uint8(Bg); 
subplot(132);
imshow(Bg);title('背景(中值滤波)');
tempI = 255 - Bg + I;
subplot(133);
imshow(tempI);title('相减的结果');
Binary = histeq(tempI);  % 直方图均衡化
figure(); subplot(142); imshow(Binary); title('直方图均衡化'); % 显示二值化图像
Binary = adapthisteq(tempI); % 自适应直方图均衡化
subplot(143); imshow(Binary); title('自适应直方图均衡化'); % 显示二值化图像
Binary = imbinarize(I, AdaptT); % 直接二值化
subplot(141); imshow(Binary); title('直接使用自适应阈值'); % 显示二值化图像
Binary = im2double(tempI).^16;   % 拉伸
subplot(144); imshow(Binary); title('拉伸'); % 显示二值化图像

h_point = detectHarrisFeatures(Binary); % Harris角点检测

condi_roi = regionprops(im2bw(1.0 - Binary), 'area', 'boundingbox'); % 求roi
max_area = 0;
index_k = 0;
figure(); subplot(131); imshow(Binary); title('二值化');
subplot(132); imshow(Binary); title('ROI与角点检测'); hold on;
for i=1:length(condi_roi)
    area = condi_roi(i).Area;
    if area > max_area
        max_area = area;
        index_k = i;
    end
end
rectangle('position', condi_roi(index_k).BoundingBox, 'EdgeColor', 'r', 'lineWidth', 1);  

plot(h_point);
hold off;

ROI = imcrop(Binary, condi_roi(index_k).BoundingBox);
tp_point = detectHarrisFeatures(ROI);
% figure(); imshow(ROI); hold on;
% plot(tp_point);
% hold off;

% % 寻找边缘四个顶点(我的方法)
% roi_pos = uint16(condi_roi(index_k).BoundingBox);
% roi_mpt = zeros(size(ROI));
% for i = 1:roi_pos(3)
%     for j = 1:10
%         roi_mpt(j, i) = 1;
%     end
% end
% for i = 1:roi_pos(3)
%     for j = -10:0
%         roi_mpt(roi_pos(4) + j, i) = 1;
%     end
% end
% for i = 1:roi_pos(4)
%     for j = 1:10
%         roi_mpt(i, j) = 1;
%     end
% end
% for i = 1:roi_pos(4)
%     for j = -10:0
%         roi_mpt(i, roi_pos(3) + j) = 1;
%     end
% end
% %figure();imshow(roi_mpt);
% 
% centerX = (roi_pos(1) + roi_pos(3))/2; % 中心点的x坐标
% centerY = (roi_pos(2) + roi_pos(4))/2; % 中心点的y坐标
% vis = zeros(1, M, 'uint16');     % 遍历相差记录
% for i=1:length(tp_point)
%     temp_pos = uint16(tp_point(i).Location);
%     if roi_mpt(temp_pos(2), temp_pos(1)) == 0
%         continue;
%     end
%     if temp_pos(1) < centerX && temp_pos(2) < centerY
%         if abs(temp_pos(1) - roi_pos(1)) < abs(temp_pos(2) - roi_pos(2))    % y轴变化了k
%             k = abs(temp_pos(2) - roi_pos(2));
%         else 
%             k = abs(temp_pos(1) - roi_pos(1));
%         end
%     elseif temp_pos(1) > centerX && temp_pos(2) < centerY
%         if abs(temp_pos(1) - roi_pos(1) - roi_pos(3)) < abs(temp_pos(2) - roi_pos(2))
%             k = abs(temp_pos(2) - roi_pos(2));
%         else 
%             k = abs(temp_pos(1) - roi_pos(1) - roi_pos(3));
%         end
% 
%     elseif temp_pos(1) < centerX && temp_pos(2) > centerY
%         if abs(temp_pos(1) - roi_pos(1)) < abs(temp_pos(2) - roi_pos(2) - roi_pos(4))
%             k = abs(temp_pos(2) - roi_pos(2) - roi_pos(4));
%         else 
%             k = abs(temp_pos(1) - roi_pos(1));
%         end
%     elseif temp_pos(1) > centerX && temp_pos(2) > centerY
%         if abs(temp_pos(1) - roi_pos(1) - roi_pos(3)) < abs(temp_pos(2) - roi_pos(2) - roi_pos(4))
%             k = abs(temp_pos(2) - roi_pos(2) - roi_pos(4));
%         else 
%             k = abs(temp_pos(1) - roi_pos(1) - roi_pos(3));
%         end
%     end
%     if k == 0
%         continue;
%     end
%     vis(k) = vis(k) + 1;
%     if vis(k) >= 2
%         if temp_pos(1) < centerX && temp_pos(2) < centerY
%             if abs(temp_pos(1) - roi_pos(1)) < abs(temp_pos(2) - roi_pos(2))    % y轴变化了k
%                 swi = 1;
%             else 
%                 swi = 0;
%             end
%         elseif temp_pos(1) > centerX && temp_pos(2) < centerY
%             if abs(temp_pos(1) - roi_pos(1) - roi_pos(3)) < abs(temp_pos(2) - roi_pos(2))
%                 swi = 0;
%             else
%                 swi = 1;
%             end
%         elseif temp_pos(1) < centerX && temp_pos(2) > centerY
%             if abs(temp_pos(1) - roi_pos(1)) < abs(temp_pos(2) - roi_pos(2) - roi_pos(4))
%                 swi = 0;
%             else 
%                 swi = 1;
%             end
%         elseif temp_pos(1) > centerX && temp_pos(2) > centerY
%            	if abs(temp_pos(1) - roi_pos(1) - roi_pos(3)) < abs(temp_pos(2) - roi_pos(2) - roi_pos(4))
%                 swi = 1;
%             else 
%                 swi = 0;
%             end
%         end
%         if swi == 1
%             dot(1, 1) = roi_pos(1) + k;                 dot(1, 2) = roi_pos(2);
%             dot(2, 1) = roi_pos(1) + roi_pos(3);        dot(2, 2) = roi_pos(2) + k;
%             dot(3, 1) = roi_pos(1);                     dot(3, 2) = roi_pos(2) + roi_pos(4) - k;
%             dot(4, 1) = roi_pos(1) + roi_pos(3) - k;    dot(4, 2) = roi_pos(2) + roi_pos(4);
%         else
%             dot(1, 1) = roi_pos(1);                     dot(1, 2) = roi_pos(2) + k;
%             dot(2, 1) = roi_pos(1) + roi_pos(3) - k;    dot(2, 2) = roi_pos(2);
%             dot(3, 1) = roi_pos(1) + k;                 dot(3, 2) = roi_pos(2) + roi_pos(4);
%             dot(4, 1) = roi_pos(1) + roi_pos(3);        dot(4, 2) = roi_pos(2) + roi_pos(4) - k;
%         end
%         break;
%     end
% end



%% 图像校正
%dot = ginput(4);
dot = double(dot);
roi_pos = double(roi_pos);
w=round(sqrt((dot(1,1)-dot(2,1))^2+(dot(1,2)-dot(2,2))^2));     %从原四边形获得新矩形宽
h=round(sqrt((dot(1,1)-dot(3,1))^2+(dot(1,2)-dot(3,2))^2));     %从原四边形获得新矩形高
w = roi_pos(3);
h = roi_pos(4);

y=[dot(1,1) dot(2,1) dot(3,1) dot(4,1)];        %四个原顶点
x=[dot(1,2) dot(2,2) dot(3,2) dot(4,2)];

%这里是新的顶点，我取的矩形,也可以做成其他的形状
%大可以原图像是矩形，新图像是从dot中取得的点组成的任意四边形.:)
Y=[dot(1,1) dot(1,1) dot(1,1)+h dot(1,1)+h];     
X=[dot(1,2) dot(1,2)+w dot(1,2) dot(1,2)+w];
Y = [roi_pos(1), roi_pos(1), roi_pos(1) + h, roi_pos(1) + h];
X = [roi_pos(2), roi_pos(2) + w, roi_pos(2), roi_pos(2) + w];

B=[X(1) Y(1) X(2) Y(2) X(3) Y(3) X(4) Y(4)]';   %变换后的四个顶点，方程右边的值
%联立解方程组，方程的系数
A=[x(1) y(1) 1 0 0 0 -X(1)*x(1) -X(1)*y(1);             
   0 0 0 x(1) y(1) 1 -Y(1)*x(1) -Y(1)*y(1);
   x(2) y(2) 1 0 0 0 -X(2)*x(2) -X(2)*y(2);
   0 0 0 x(2) y(2) 1 -Y(2)*x(2) -Y(2)*y(2);
   x(3) y(3) 1 0 0 0 -X(3)*x(3) -X(3)*y(3);
   0 0 0 x(3) y(3) 1 -Y(3)*x(3) -Y(3)*y(3);
   x(4) y(4) 1 0 0 0 -X(4)*x(4) -X(4)*y(4);
   0 0 0 x(4) y(4) 1 -Y(4)*x(4) -Y(4)*y(4)];

fa=A\B;        %用四点求得的方程的解，也是全局变换系数
a=fa(1);b=fa(2);c=fa(3);
d=fa(4);e=fa(5);f=fa(6);
g=fa(7);h=fa(8);

rot=[d e f;
     a b c;
     g h 1];        %公式中第一个数是x,Matlab第一个表示y，所以我矩阵1,2行互换了

pix1=rot*[1 1 1]'/(g*1+h*1+1);  %变换后图像左上点
pix2=rot*[1 N 1]'/(g*1+h*N+1);  %变换后图像右上点
pix3=rot*[M 1 1]'/(g*M+h*1+1);  %变换后图像左下点
pix4=rot*[M N 1]'/(g*M+h*N+1);  %变换后图像右下点

height=round(max([pix1(1) pix2(1) pix3(1) pix4(1)])-min([pix1(1) pix2(1) pix3(1) pix4(1)]));     %变换后图像的高度
width=round(max([pix1(2) pix2(2) pix3(2) pix4(2)])-min([pix1(2) pix2(2) pix3(2) pix4(2)]));      %变换后图像的宽度
imgn=zeros(height,width);

delta_y=round(abs(min([pix1(1) pix2(1) pix3(1) pix4(1)])));            %取得y方向的负轴超出的偏移量
delta_x=round(abs(min([pix1(2) pix2(2) pix3(2) pix4(2)])));            %取得x方向的负轴超出的偏移量
inv_rot=inv(rot);

for i = 1-delta_y:height-delta_y                        %从变换图像中反向寻找原图像的点，以免出现空洞，和旋转放大原理一样
    for j = 1-delta_x:width-delta_x
        pix=inv_rot*[i j 1]';       %求原图像中坐标，因为[YW XW W]=fa*[y x 1],所以这里求的是[YW XW W],W=gy+hx+1;
        pix=inv([g*pix(1)-1 h*pix(1);g*pix(2) h*pix(2)-1])*[-pix(1) -pix(2)]'; %相当于解[pix(1)*(gy+hx+1) pix(2)*(gy+hx+1)]=[y x],这样一个方程，求y和x，最后pix=[y x];
        
        if pix(1)>=0.5 && pix(2)>=0.5 && pix(1)<=M && pix(2)<=N
            imgn(i+delta_y,j+delta_x)=I(round(pix(1)),round(pix(2)));     %最邻近插值,也可以用双线性或双立方插值
        end  
    end
end

subplot(133); imshow(uint8(imgn)); title('图像校正');

%figure(); imshow(Binary); hold on;
% rectangle('Position', [dot(1, 1), dot(1, 2), dot(2, 1)-dot(1,1), dot(3,2)-dot(1,2)], 'EdgeColor', 'r', 'lineWidth', 1);
% hold off;
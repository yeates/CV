%function pretraindata
filepath0 = 'digittest/训练集/'; % 读取训练集目录
filepaths=dir(filepath0);
for i=3:size(filepaths,1) % 从0到9创建目录
    filenames=dir([filepath0 filepaths(i).name '/']); % 打开当前数字的训练集目录
    % 生成训练集
    for j=3:size(filenames,1)
        img=imread([filepath0 filepaths(i).name '/'  filenames(j).name]);
        img=rgb2gray(img);
        img=imresize(img,[20,15]);
        imwrite(img,[filepath0 filepaths(i).name '/'  num2str(j-3) '-' num2str(cnt) '.bmp'])
        outputView = imref2d(size(img));
        for cnt=1:100
            tform=zeros(3);
            tform(1:2,1:2)=eye(2)-rand(2)/10;
            tform(3,3)=1;
            T= affine2d(tform);
            I= imwarp(img,T, 'OutputView', outputView);
            imwrite(I,[filepath0 filepaths(i).name '/'  num2str(j-3) '-' num2str(cnt) '.bmp'])
        end
     end
 end

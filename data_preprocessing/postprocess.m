% Modified from pix2pix scripts: https://github.com/phillipi/pix2pix/blob/master/scripts/edges

%%% Prerequisites
% You need to get the cpp file edgesNmsMex.cpp from https://raw.githubusercontent.com/pdollar/edges/master/private/edgesNmsMex.cpp
% and compile it in Matlab: mex edgesNmsMex.cpp
% You also need to download and install Piotr's Computer Vision Matlab Toolbox:  https://pdollar.github.io/toolbox/

%%% parameters
% hed_mat_dir: the hed mat file directory (the output of 'batch_hed.py')
% edge_dir: the output HED edges directory
% image_width: resize the edge map to [image_width, image_width] 
% threshold: threshold for image binarization (default 25.0/255.0)
% small_edge: remove small edges (default 5)

function [] = PostprocessHED(hed_mat_dir, edge_dir, image_width, threshold, small_edge)

if ~exist(edge_dir, 'dir')
    mkdir(edge_dir);
end
classes = dir(fullfile(hed_mat_dir, "*"));
fileList =  [];
n_classes = numel(classes);

for c = 3: n_classes
cname = classes(c).name
fileList = dir(fullfile(hed_mat_dir, cname, '*.mat'));
    %for n = 1:n_img
    %disp(dir(fullfile(hed_mat_dir, cname, '*.png')).name);
    %end
nFiles = numel(fileList);
fprintf('find %d mat files\n', nFiles);
if ~exist(fullfile(edge_dir, cname), 'dir')
    mkdir(fullfile(edge_dir, cname)); 
end
for n = 1 : nFiles
    if mod(n, 1000) == 0
        fprintf('process %d/%d images\n', n, nFiles);
    end
    fileName = fileList(n).name;
    filePath = fullfile(hed_mat_dir, cname, fileName);
    jpgName = fileName;
    jpgName = strrep(fileName, "mat", "png");
    edge_path = fullfile(edge_dir, cname, jpgName);
    
    %if ~exist(edge_path, 'file')
        E = GetEdge(filePath);
        %E = 1 - imread(filePath);
        E = imresize(E,[256,256]);
        E_simple = SimpleEdge(E, threshold, small_edge);
        E_simple = uint8(E_simple*255);
        E = imresize(E,[image_width,image_width]);
        imwrite(E_simple, edge_path);
    %end
end
end
end




function [E] = GetEdge(filePath)
matdata = load(filePath);
E = 1-matdata.predict;
end

function [E4] = SimpleEdge(E, threshold, small_edge)
if nargin <= 1
    threshold = 25.0/255.0;
end

if nargin <= 2
    small_edge = 5;
end

if ndims(E) == 3
    E = E(:,:,1);
end

E1 = 1 - E;
E2 = EdgeNMS(E1);
E3 = double(E2>=max(eps,threshold));
E3 = bwmorph(bwmorph(E3,'thin',inf), 'clean');
E4 = bwareaopen(E3, small_edge);
E4 = bwmorph(bwmorph(E4,'shrink'), 'spur');
E4=1-E4;
end

function [E_nms] = EdgeNMS( E )
E=single(E);
[Ox,Oy] = gradient2(convTri(E,4));
[Oxx,~] = gradient2(Ox);
[Oxy,Oyy] = gradient2(Oy);
O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
E_nms = edgesNmsMex(E,O,1,5,1.01,1);
end


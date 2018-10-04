clear;
clc;
%Specify directory with input data
directory = 'C:/Testdata/tsne/Dialyzer';

disp('t-SNE for the multidimensional data');
% Let's upload descriptions first
Features = csvread([directory, '/tsne_dscr.csv']);
%lets upload labels (it may seems bulky, but this method works for the latin and cyrillic symbols)
[fID,errormsg] = fopen([directory, '/tsne_lbls.txt'],'r','n','UTF-8');
SL = textscan(fID,'%[^\n]','delimiter','\n');
Labels = SL{1};

% Now we can make t-SNE
outDims = 2; pcaDims = size(Features,2); perplexity = 30; theta = 0.1; alg = 'svd';
disp('Data has been prepared successfully. t-SNE started...');
map = fast_tsne(Features, outDims, pcaDims, perplexity, theta, alg, 5000); % many thanks to https://github.com/lvdmaaten/bhtsne

disp('Prepare plot data...');   
fig = figure
    coords = zeros(size(map));
    maxc = max(map);
    minc = min(map);
    a(1,1) = 1.0 / (maxc(1,1) - minc(1,1));
    b(1,1) = - minc(1,1) * a(1,1);
    a(1,2) = 1.0 / (maxc(1,2) - minc(1,2));
    b(1,2) = - minc(1,2) * a(1,2);
    for i = 1:size(coords,1)
        coords(i,1) = 0.9*(map(i,1)*a(1,1) + b(1,1)) + 0.05;
        coords(i,2) = 0.9*(map(i,2)*a(1,2) + b(1,2)) + 0.05;
    end
    for i = 1:size(Labels,1)
     axes('pos',[coords(i,1), coords(i,2), 0.045, 0.045]); % this params control how small thumbnails will be
     try
        I = imresize(imread(Labels{i}),[300 400]);
        imshow(I);
     catch
        warning(['Can not load ' Labels{i}]); 
     end
    end
   
disp('Saving plot on hard drive as png file...'); 
set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0 0 1920 1080]/800);
print(fig, [directory, '/tsne_graph.png'], '-dpng', '-r800');
close(fig);
disp('Work has been finished');
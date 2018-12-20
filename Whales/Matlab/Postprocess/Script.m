clear;
clc;
disp('Please wait...');
listoffiles = ls('submissions/*.csv');
data{1,1,1} = '';
for i = 1:size(listoffiles,1)
    if i == 1
        data = table2array(readtable(['submissions/',deblank(listoffiles(i,:))],'ReadVariableNames',false));     
    else
        data = cat(3,data,table2array(readtable(['submissions/',deblank(listoffiles(i,:))],'ReadVariableNames',false)));
    end
end

output{size(data,1),size(data,2)} = '';
for i = 1:size(data,1)
    output{i,1} = data{i,1,1};
    for j = 2:size(data,2)
        output{i,j} = mostWanted(data(i,j,:)); 
    end
end

file = fopen('postprocessed.csv','w');
fprintf(file,'Image,Id\n');
for i = 1:size(output,1)
    fprintf(file,[output{i,1},output{i,2},' ',output{i,3},' ',output{i,4},' ',output{i,5},' ',output{i,6},'\n']);
end
fclose(file);
disp('Done');

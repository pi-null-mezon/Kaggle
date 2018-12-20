function [name] = mostWanted(C)
% input should be cell array with {1,1,:} structure
Votes{size(C,3)} = '';
for i = 1:size(C,3)
    Votes{i} = C{1,1,i};
end
[uniqueVotes, ~, J] = unique(Votes);
[Val,idx] = max(histcounts(J));
name = uniqueVotes{idx};
end


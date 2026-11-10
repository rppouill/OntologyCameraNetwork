function [predicted, distance] = imageClassify(w1, w2, metric, params)
    if nargin < 3 || isempty(metric)
        metric = 'spectral';
    end
    if nargin < 4
        params = struct();
    end
    predicted = zeros(size(w2,1),1);
    distance  = zeros(size(w2,2),1);
    for i = 1:size(w1, 2)
        mini = Inf;
        for j = 1:size(w2, 1)
            dist = computeDistance(w1{i}, w2{j}, metric, params);
            if dist < mini
                mini = dist;
                predicted(i) = j;
                distance(i) = dist;
            end
        end        
    end
end
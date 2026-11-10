function [predicted, distance] = imageClassify(w1, w2)
    predicted = zeros(size(w2,1),1);
    distance  = zeros(size(w2,2),1);
    for i = 1:size(w1, 2)
        mini = Inf;
        for j = 1:size(w2, 1)
            dist = norm(w1{i} - w2{j});
            if dist < mini
                mini = dist;
                predicted(i) = j;
                distance(i) = dist;
            end
        end        
    end
end
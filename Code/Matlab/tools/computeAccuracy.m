function accuracy = computeAccuracy(predict)
    accuracy = 0;
    for i = 1:size(predict, 1)
        if predict(i) == i
            accuracy = accuracy + 1;
        end
    end
    accuracy = accuracy / size(predict, 1);
end
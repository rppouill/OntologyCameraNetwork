function accuracy = evaluate(input, output, X, show)
    if nargin < 4
        show = false;
    end

    predict = cell(size(input,2),1);
    for k = 1:size(input,2)
        predict{k} = X * input{k};
    end

    [class, ~] = imageClassify(output,predict);
    accuracy = computeAccuracy(class);

    if show
        figure("Name", "Predictions");
        %set(gcf, 'Units', 'normalized', 'Position', [0 0 1 1]);
        for i = 1:size(input,2)
            subplot(3,size(input,2),i);                     imshow(mat2gray(reshape(input{i}(:,1)   ,[15,15]))); title(['Person: ', num2str(i)]);
            subplot(3,size(input,2),i + size(input,2));     imshow(mat2gray(reshape(output{i}(:,1)  ,[15,15])));
            subplot(3,size(input,2),i + 2*size(input,2));   imshow(mat2gray(reshape(predict{i}(:,1) ,[15,15])));
            % Plot prediction
            if class(i) == i
                title(['Predicted: ', num2str(class(i)), ' | Real: ', num2str(i)]);
            else
                title(['Predicted: ', num2str(class(i)), ' | Real: ', num2str(i)], 'Color', 'red');
            end 
        end
    end

end
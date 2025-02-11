function fisherAxes = train2DLDA(trainingData, d)
    % trainingData  : Cell array of training data
     L          = size(trainingData, 1);
    [m, n, M]   = size(trainingData{1});


    if nargin < 2
        d = L;
    end
    
    A_Global = zeros(m,n);
    for i = 1:L
        A_Global = A_Global + sum(trainingData{i}, 3);
    end
    A_Global = A_Global / M;
    
    S_w = zeros(m,n);
    S_b = zeros(m,n);
    for i = 1:L

        N_i = size(trainingData{i}, 3);
        A_i = mean(trainingData{i}, 3);

        diff_B = A_i - A_Global;
        S_b = S_b +  N_i * (diff_B' * diff_B);

        for j = 1:N_i
            diff_W = trainingData{i}(:,:,j) - A_i;
            S_w = S_w + diff_W' * diff_W;
        end
    end


    % Compute the transformation matrix
    [V, D] = Turk_Pentland(pinv(S_w) * S_b, d);
    fisherAxes = D;

end


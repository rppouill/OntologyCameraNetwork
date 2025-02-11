function [pcaEigVal,pcaEigVec] = Turk_Pentland(data, num_components)
    L = transpose(data) * data; % Turk-Pentlant trick (part 1)
    [eigVec, eigVal] = eig(L);

    diag_eigVal = diag(eigVal);
    [~, index] = sort(diag_eigVal);
    index = flipud(index);

    pcaEigVal = zeros(size(eigVal));
    pcaEigVec = zeros(size(eigVec));

    for k = 1:num_components
        pcaEigVal(k,k) = eigVal(index(k),index(k));
        pcaEigVec(:,k) = eigVec(:,index(k));
    end

    pcaEigVal = diag(pcaEigVal);
    pcaEigVal = pcaEigVal / 2;
    pcaEigVal = pcaEigVal(1:num_components);

    pcaEigVec = data * pcaEigVec; % Turk-Pentlant trick (part 2)

    % Normalizing the eigenvectors
    for k = 1:num_components
        pcaEigVec(:,k) = pcaEigVec(:,k) / norm(pcaEigVec(:,k));
    end

    % Creating lower dimensional subspace
    pcaEigVec = pcaEigVec(:, 1:num_components);
    pcaEigVec = round(pcaEigVec,4);
end
    
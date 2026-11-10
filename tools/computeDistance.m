function d = computeDistance(A, B, metric, params)
    % COMPUTEDISTANCE Distance between two eigenvector matrices
    %   d = COMPUTEDISTANCE(A, B)                 spectral norm of A - B (original behaviour)
    %   d = COMPUTEDISTANCE(A, B, metric)
    %   d = COMPUTEDISTANCE(A, B, metric, params)
    %
    %   A, B   : matrices of the same size (D x nVector), one eigenvector per column
    %   metric : 'spectral'    - norm(A - B), largest singular value (default, = original code)
    %            'euclidean'   - Frobenius norm of A - B
    %            'seuclidean'  - standardized Euclidean, needs params.var (D x 1)
    %                            = per-dimension variance of the prediction residuals
    %            'mahalanobis' - needs params.SigmaInv (D x D) or params.Sigma (D x D)
    %                            = (regularized) covariance of the prediction residuals
    %            'cosine'      - 1 - |<A,B>| / (||A|| ||B||)
    %            'subspace'    - geodesic distance between span(A) and span(B)
    %                            (principal angles)

    if nargin < 3 || isempty(metric)
        metric = 'spectral';
    end
    if nargin < 4
        params = struct();
    end

    switch lower(metric)
        case 'spectral'
            d = norm(A - B);

        case 'euclidean'
            d = norm(A - B, 'fro');

        case 'seuclidean'
            if ~isfield(params, 'var')
                error('computeDistance: params.var is required for seuclidean');
            end
            R = A - B;
            d = sqrt(sum((R.^2) ./ params.var(:), 'all'));

        case 'mahalanobis'
            R = A - B;
            if isfield(params, 'SigmaInv')
                SR = params.SigmaInv * R;
            elseif isfield(params, 'Sigma')
                SR = params.Sigma \ R;
            else
                error('computeDistance: params.SigmaInv or params.Sigma is required for mahalanobis');
            end
            d = sqrt(max(0, sum(R .* SR, 'all')));

        case 'cosine'
            nA = norm(A, 'fro'); nB = norm(B, 'fro');
            if nA == 0 || nB == 0
                d = 1;
            else
                d = 1 - abs(sum(A .* B, 'all')) / (nA * nB);
            end

        case 'subspace'
            QA = orth(A); QB = orth(B);
            s  = svd(QA' * QB);
            s  = min(1, max(0, s));
            d  = sqrt(sum(acos(s).^2));

        otherwise
            error('computeDistance: unknown metric "%s"', metric);
    end
end

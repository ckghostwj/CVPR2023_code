function F = solveF(Z,numOfClasses)

numOfViews = length(Z);

sumLZ = 0;
for i=1:numOfViews
    W = 0.5*(Z{i}+Z{i}');    
    M = diag(sum(W))-W;
    M(isnan(M))=0;
    M(isinf(M))=1e5;
    sumLZ = sumLZ+M;
end

try
    [V,D] = eig(sumLZ);
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
        [V,D] = eig(sumLZ, eye(size(sumLZ)));
    else
        rethrow(ME);
    end
end
[D_sort, ind] = sort(diag(D));
ind2 = find(D_sort>1e-6);
if length(ind2)>numOfClasses
    F = V(:, ind2(1:numOfClasses));
else
    F = V(:, ind(1:numOfClasses));
end


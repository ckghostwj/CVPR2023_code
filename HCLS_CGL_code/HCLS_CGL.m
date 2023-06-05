function [F,Z,obj] = HCLS_CGL(Skiv,Wiv,XWiv,F_ini,num_clust,lambda2,lambda3,para_r,max_iter)

num_view = length(Skiv);

F = F_ini;
alpha = ones(1,num_view);
alpha_r = alpha.^para_r;
Z_linshi=  0;
W_linshi = 0;
for iv = 1:num_view
    Z_linshi = Z_linshi+Skiv{iv}.*XWiv{iv};
    W_linshi = W_linshi+XWiv{iv};
    Wiv2{iv} = Wiv{iv}.*Wiv{iv};  % H^(v).*H^(v) in paper
end
Z = Z_linshi./(W_linshi+eps);
Z(isnan(Z)) = 0;
Z(isinf(Z)) = 1;
for iter = 1:max_iter

    % ------------ Z -------------- %
    linshi_fenzi = 0;
    linshi_fenmu = 0;
    for iv = 1:num_view
        linshi_fenzi = linshi_fenzi + alpha_r(iv)*Skiv{iv}.*Wiv2{iv};
        linshi_fenmu = linshi_fenmu + alpha_r(iv)*Wiv2{iv};
    end
    linshi_P = EuDist2(F,F,0);  % E^(v) in eq.(10)/(11)
    linshi_P = linshi_P - diag(diag(linshi_P));
    linshi_Z = (linshi_fenzi-0.25*lambda3*linshi_P)./(linshi_fenmu+lambda2); % T in eq.(12)
    Z = zeros(size(linshi_Z));
    for in = 1:size(Z,2)
        linshi_c = 1:size(linshi_Z,1);
        linshi_c(in) = [];
        Z(linshi_c,in) = EProjSimplex_new(linshi_Z(linshi_c,in));      % S in eq.(12)  
    end   
    % ----------- F ------ %
    linshiZ = (Z+Z')*0.5;
    LapZ = diag(sum(linshiZ))-linshiZ;
    [F, ~,~] = eig1(LapZ, num_clust, 0);  % solving problem (13)
    % ------- alpha ------- %
    for iv = 1:num_view
       Rec_error(iv) = norm((Z-Skiv{iv}).*Wiv{iv},'fro')^2;
    end
    linshi_H = bsxfun(@power,Rec_error, 1/(1-para_r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,linshi_H,sum(linshi_H));         % eq.(17)
    alpha_r = alpha.^para_r;  
    % ----------------- obj ----------- %
    linshiZ = (Z+Z')*0.5;
    LapZ = diag(sum(linshiZ))-linshiZ;  
    obj(iter) = alpha_r*Rec_error'+lambda3*trace(F'*LapZ*F)+lambda2*norm(Z,'fro')^2;
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-6
        iter
        break;
    end
end

end
function Ys = TSLRLN_fast(Y, lambda, gamma, sub_dim, iter)
F=FastHyDe(Y, 'additive', 0, sub_dim);
F=F.*255;Y=Y.*255;
Y_new = permute(Y,[2,3,1]);F_new= permute(F,[2,3,1]);
[U,Z,V]=tsvd(F_new);
S_est=zeros(size(Y_new));

D_est(:,1:sub_dim,:) = U(:,1:sub_dim,:);
Z_est(1:sub_dim,1:sub_dim,:) = Z(1:sub_dim,1:sub_dim,:);
V_est(:,1:sub_dim,:) = V(:,1:sub_dim,:);

C_est= tprod(Z_est,tran(V_est));
% C_res=C_est;
Y_rec=tprod(D_est,C_est);
iter1=1;
while iter1 <= iter
    iter2 = iter1;
    %     while ( norm(C_est(:)-C_res(:),'fro') / norm(C_res(:),'fro') ) >= 10e-3
    while iter2 <= 10
        C_res = C_est;
        C_nl = tprod(tran(D_est),(Y_new-S_est));
        C_nl = permute(C_nl,[3,2,1]);
        C_nl = C_nl(:,:,1:sub_dim);
        % denoising the variable Z with BM4D filter
        C_max = max(C_nl(:));
        C_min = min(C_nl(:));
        C_nl = (C_nl-C_min)./(C_max-C_min); % scale the data to [0 -255]
        [C_nl, ~] = bm4d(C_nl, 'Gauss', 0);
        C_nl = C_nl.*(C_max-C_min) + C_min;
        temp(:,:,1:sub_dim) = C_nl;
        C_est=ipermute(temp,[3,2,1]);
        
        [U2,~,V2] = tsvd(tprod(C_est,tran(Y_new-S_est)));
        D_est = tprod(V2,tran(U2));
        
        S_est = Y_new-tprod(D_est,C_est);
        Weight = 1./(abs(S_est)+eps);
        S_est = prox_l1((S_est),Weight*lambda);
        
        Y_new = tprod(D_est,C_est);
        Y_new = prox_tnn(Y_new,1/gamma);
        
        if ( norm(C_est(:)-C_res(:),'fro') / norm(C_res(:),'fro') ) <= 10e-2
            break;
        end
        iter2 = iter2 + 1;
    end
    if iter1 < iter
        Y_new = Y_rec;
    end
    iter1 = iter1 + 1;
end
Ys = ipermute(Y_new,[2,3,1]);
Ys = Ys./255;
end
function [indexPerFold] = obtenirListeParFold(Y,nFolds)

% X est un prdataset
% nFolds est le nombre de folds désiré
% La fonction retourne une structure contenant des listes 
% d'index. Il y a une liste d'index par fold. 
indexPerFold = cell(nFolds,1);

idx = 1:size(Y,1);

% obtenir la liste de toutes les classes
Ynames = unique(Y);

ctr = 0;
for c = 1:length(Ynames) 
    
    tmpI = idx(Y==Ynames(c));
    rp = randperm(length(tmpI));
    tmpI = tmpI(rp);
    
    for i = 1:length(tmpI)
        f = mod(ctr,nFolds)+1;
        indexPerFold{f} = [indexPerFold{f};tmpI(i)];
        ctr = ctr+1;
    end
    
end

end
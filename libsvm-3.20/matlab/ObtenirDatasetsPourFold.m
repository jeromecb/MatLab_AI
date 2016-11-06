function [XTR, YTR, XVAL, YVAL] = ObtenirDatasetsPourFold(X,Y,indexPerFold,valFoldNumber)
% Cette fonction retourne deux prdataset un pour la validation (VAL) 
% et l'autre pour l'entrainement TR

% X : toutes les données dans une  matrice
% Y : les labels correspondant à X
% indexPerFold : est une structure de liste d'index obtenue avec la 
% fonction obtenirListeParFold
% valFoldNumber: Est le numéro du fold utilisé pour la validation


idxTR = [];

for f = 1:size(indexPerFold,1) 

    if f == valFoldNumber
        idxVal = indexPerFold{f};
    else        
        idxTR = [idxTR;indexPerFold{f}];
    end  
end

XTR = X(idxTR,:);
XVAL = X(idxVal,:);

YTR = Y(idxTR,:);
YVAL = Y(idxVal,:);

end
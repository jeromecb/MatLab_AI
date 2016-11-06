%% LAB 3
clear all
close all
clc
%% Charger LIBSVM
addpath(genpath('libsvm-3.20\matlab\'))

%% Charger la base de données
load Prob2.mat

%% Apprendre avec le SVM (le validation croisée n'est pas implémentée)
nFolds = 10;
f = 3;
C = [0.01 0.1 1 10 100 1000];

gamma = [0.0001 0.001 0.01 0.1 1 10 100];

kernel = 2;

% séparer les données en folds
[indexPerFold] = obtenirListeParFold(Ytrain,nFolds);

%Calcul du temps de traitement
accT = [];

% obtenir les données d'entrainement et de validation selon
% la sépration que l'on vient d'effectuer. 
[XTR, YTR, XVAL, YVAL] = ObtenirDatasetsPourFold(Xtrain,Ytrain,indexPerFold,f);

%%Entrainement supervisé
%Ici on va venir faire l'entrainement superviser de notre SVM

for g=1:length(gamma)
    for c=1:length(C)
        for f=1:nFolds      
            % créer un string contenant les paramètres pour configurer le SVM
            % pout plus d'information taper svmtrain dans la console.
            SVMstr = ['-t ' num2str(kernel) ' -c ' num2str(C(c)) ' -g ' ...
                num2str(gamma(g)) ' -q'];
            % obtenir les données d'entrainement et de validation selon
            % la sépration que l'on vient d'effectuer.
            [XTR, YTR, XVAL, YVAL] = ObtenirDatasetsPourFold(Xtrain,Ytrain,indexPerFold,f);
            
            % Entrainer le SVM sur la base d'entrainement
            modelSVMGAUSS = svmtrain(YTR,XTR,SVMstr);
            
            % classifier la base de validation avec le SVM
            [predY, accuracySVMGAUSS(f,:), DV] = svmpredict(YVAL, XVAL, modelSVMGAUSS, '-q');
          
        end
        %Accumulation des taux en 3D gamma et C
        accTauxGAUSS(g,c) = mean(accuracySVMGAUSS(:,1)); 
    end
end

%%Comme on a pu le voir dans le papier de l'ÉTS ici on va venir libeler nos
%%valeurs encore non libelé pour pouvoir apprendre avec ces valeurs par la
%%suite

accuracyGAUSS = max(accTauxGAUSS);
[rowGAUSS colGAUSS] = find(accTauxGAUSS == max(max(accTauxGAUSS)));

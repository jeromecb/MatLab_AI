%% LAB 3
clear all
close all
clc
%% Charger LIBSVM
addpath(genpath('libsvm-3.20\matlab\'))

%% Charger la base de donn�es
load Prob2.mat

%% Apprendre avec le SVM (le validation crois�e n'est pas impl�ment�e)
nFolds = 10;
f = 3;
C = [0.01 0.1 1 10 100 1000];

gamma = [0.0001 0.001 0.01 0.1 1 10 100];

kernel = 0;

% s�parer les donn�es en folds
[indexPerFold] = obtenirListeParFold(Ytrain,nFolds);

%Calcul du temps de traitement
accT = [];

% obtenir les donn�es d'entrainement et de validation selon
% la s�pration que l'on vient d'effectuer. 
[XTR, YTR, XVAL, YVAL] = ObtenirDatasetsPourFold(Xtrain,Ytrain,indexPerFold,f);


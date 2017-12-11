clc;clear;close all;
addpath('./Functions');
load ./sample_image.mat sample_image;
figure;
imagesc(sample_image);
colormap(jet);
%%
x = (1:size(sample_image,2))';
y = (1:size(sample_image,1))';
order = 1;
Element = constructElements(x,y,0,order);
%%
Mset=(unique(sample_image(:)))';
beta_ini = mvnrnd(zeros(1,4),diag(0.1*ones(4,1)))'; % beta is a column vector
SigmaProp_ini = diag(0.01*ones(4,1));
MC_ini = retrieve(Element,sample_image);
figure;
plotField(Element,MC_ini,jet);
%%
Chain_length = 500;

Element = FixElement(Element,MC_ini);
Element = CalculateU(Element,zeros(1,length(Mset)));
Element = detectNeighborDirection(Element,2);

[MC_est,U_bin,beta_bin]=GenerateMRF(Element,MC_ini,Mset,Chain_length,beta_ini,SigmaProp_ini);    

plot(1:Chain_length,beta_bin);

%%
U_chain = nansum(U_bin);
figure;
plot(2:length(U_chain),U_chain(2:end));
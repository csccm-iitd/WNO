clear
close all
clc

L = 1;
x = linspace(-L, L, 512);
t = linspace(0, 1, 51);
sample = 500;
a = 0;
b = 0.5;
r = (b-a).*rand(sample,1) + a;

global np
rng(1)

sol = zeros(sample,512,51);
m=0;
% np=0;
% sol1 = pdepe(m,@burgerpde,@burgeric,@burgerbc,x,t);
% surf(sol1)

for i=1:sample
    i
    np = r(i); 
    sol1 = pdepe(m,@burgerpde,@burgeric,@burgerbc,x,t);
    sol(i,:,:) = sol1';
end

%%
figure;
index = 1;
for i=1:sample
    if mod(i,50)==0
        i
        subplot(2,5,index); imagesc(squeeze(sol(i,:,:)));
        index = index+1;
    end
end
 
%%
figure; surf(squeeze(sol(499,:,:)));

save ('burgers_data_512_51.mat', 't', 'x', 'sol' )

function [c,f,s] = burgerpde(x,t,u,dudx)
c = 1;
f = (0.01/pi)*dudx;
s = -u*dudx;
end
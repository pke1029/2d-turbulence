% pseudo spectral method for 3D Euler flow equation
clear, clc,
global dx dy kx ky k2 alias

% space grid
Nx = 128;
Ny = 128;
Lx = 2*pi;
Ly = 2*pi;
dx = Lx / Nx;
dy = Ly / Ny;
x = (0:Nx-1)*dx;
y = (0:Ny-1)*dy;

% time grid
dt = 0.001;
T = 1.1;
t = 0:dt:T;
Nt = length(t);

% frequency grid
kx = ifftshift((2*pi/Lx) * (-Nx/2:Nx/2-1));
ky = ifftshift((2*pi/Ly) * (-Ny/2:Ny/2-1));
kmax = min(max(kx), max(ky));
k2 = kx'.^2 + ky.^2;
k = sqrt(k2);
alias = k > 0.66*kmax;       % dealiasing array

% initial condition
w = -sin(x') - cos(x').*cos(y); 
g = sin(x').*sin(y) - cos(y);

% configure plot
figure
set(pcolor(x, y, w), 'edgecolor', 'none')
colormap hot
axis equal
xlim([x(1), x(end)])
ylim([y(1), y(end)])
colorbar
hold on

C = zeros(1, Nt-1);
n = zeros(1, Nt-1);
delta = zeros(1, Nt-1);
infg = zeros(1, Nt-1);

for i = 2:Nt
    
    % energy spectrum
    g_hat = fft2(g);
    [E, C(i-1), n(i-1), delta(i-1)] = energySpectrum(k, round(0.66*kmax), g_hat);
    
    % infimum of gamma
    infg(i-1) = min(g, [], 'all');
    
    % Runge-Kutta
    [w1, g1] = ODE(w, g);
    [w2, g2] = ODE(w + w1*dt/2, g + g1*dt/2);
    [w3, g3] = ODE(w + w2*dt/2, g + g2*dt/2);
    [w4, g4] = ODE(w + w3*dt, g + g3*dt);
    
    w = w + (dt/6) * (w1 + 2*w2 + 2*w3 + w4);
    g = g + (dt/6) * (g1 + 2*g2 + 2*g3 + g4);
    
    if ~isfinite(w) | ~isfinite(g); break; end
    
    % show
    if mod(i, 100) == 1
        set(pcolor(x, y, w), 'edgecolor', 'none')
        title(sprintf('t = %0.3f',t(i)))
        colorbar
        drawnow
    end
    
end

hold off
figure
subplot(3,1,1)
semilogy(t(1:end-1), delta, 'o')
xlim([0 , 1.1])
subplot(3,1,2)
plot(t(1:end-1), n, 'o')
xlim([0 , 1.1])
subplot(3,1,3)
semilogy(t(2:end-1), C(2:end)./infg(2:end).^2, 'o')
xlim([0 , 1.1])

function [dwdt, dgdt] = ODE(w, g)

global dx dy kx ky k2 alias

% fourier transform
w_hat = fft2(w);
g_hat = fft2(g);

% dealiasing 
w_hat(alias) = 0;
g_hat(alias) = 0;

% velocity field
u_hat = 1j*(kx'.*g_hat+ky.*w_hat)./k2;
v_hat = 1j*(ky.*g_hat-kx'.*w_hat)./k2;

% no zero mode
u_hat(1,1) = 0;
v_hat(1,1) = 0;

% inverse transform
u = real(ifft2(u_hat));
v = real(ifft2(v_hat));

% average gamma^2 
avg_g2 = 1/(4*pi^2)*sum(g.^2, 'all')*dx*dy;

% differentiation in fourier space
wx = real(ifft2(1j*kx'.*w_hat));     % dw/dx
wy = real(ifft2(1j*ky.*w_hat));      % dw/dy
gx = real(ifft2(1j*kx'.*g_hat));     % dg/dx
gy = real(ifft2(1j*ky.*g_hat));      % dg/dy

% ODE
dwdt = g.*w - u.*wx - v.*wy;              % dw/dt
dgdt = 2*avg_g2 - g.^2 - u.*gx - v.*gy;   % dg/dt

end

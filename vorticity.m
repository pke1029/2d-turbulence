%{
      ~ Spectral method for 2D Navier-Stokes (w/ periodic BC) ~

We investigate the problem given by
        dtW + C = nu * (dxxW + dyyW) + F

Taking the Fourier transform of the equation becomes
        dtW_hat + C_hat = - nu * |k|^2 * W_hat + F_hat

Discretise W_hat with Crank-Nicolson, C_hat with Adam Bashforth.

%}

% space grid
Nx = 128;
Ny = 128;
Lx = 2*pi;
Ly = 2*pi;
dx = Lx / Nx;
dy = Ly / Ny;
x = 0:dx:Lx-dx;
y = 0:dy:Ly-dy;

% time grid
dt = 0.001;
T = 50;
t = 0:dt:T;
Nt = length(t);

% simulation parameters
kmin = 7;           % lower wave number
kmax = 9;           % upper wave number 
nu = 0.00025;       % viscosity coeff
% nu = 5.9e-30;
R = 0.9;

% frequency grid
kx = (2*pi/Lx) * (-Nx/2:Nx/2-1);
ky = (2*pi/Ly) * (-Ny/2:Ny/2-1);
k = sqrt(kx'.^2 + ky.^2);

% inverse Laplace
invL = k.^2;
invL(Nx/2+1, Ny/2+1) = 1;
invL = 1./invL;

% initial condition
w = gauss2d(x-pi, y-pi*0.75, 0.2) + gauss2d(x-pi, y-pi*5/4, 0.2);
w_hat = fftshift(fft2(w));

% forcing
F_hat = zeros(Nx, Ny);
A = Nx*Ny/sqrt(pi*(kmax^2-kmin^2))*sqrt(2);     % normalization factor

% initialize C_hat
C_hat0 = zeros(Nx, Ny);
C_hat1 = zeros(Nx, Ny);
C_hat2 = zeros(Nx, Ny);

for i = 2:Nt
    
    C_hat0 = C_hat1;
    C_hat1 = C_hat2;
    
    u = real(ifft2(ifftshift(-1j*ky.*invL.*w_hat)));    % x velocity
    v = real(ifft2(ifftshift(1j*kx'.*invL.*w_hat)));    % y velocity
    
    wx = real(ifft2(ifftshift(1j*kx'.*w_hat)));         % dw/dx
    wy = real(ifft2(ifftshift(1j*ky.*w_hat)));          % dw/dy
    
    % convective term
    C = u.*wx + v.*wy;
    C_hat2 = fftshift(fft2(C));
    C_hat2(k > 0.66*Nx/2) = 0;      % aliasing
    
    if i < 4; C_hat = C_hat2;                                       % Euler (1st order)
    else; C_hat = 23/12*C_hat2 - 16/12*C_hat1 + 5/12*C_hat0; end    % Adam Bashforth (3rd order)
    
    % forcing term
    randF_hat = A * exp(2*pi*1j*rand(Nx, Ny));
    randF_hat(k < kmin | k > kmax) = 0;             % scale of forcing
    F_hat = R*F_hat + sqrt(1-R^2).*randF_hat;
    
    % crank-nicolson
    w_hat = (w_hat.*(1-0.5*nu*dt*k.^2) - dt*C_hat + dt*F_hat) ./ (1+0.5*nu*dt*k.^2);
    w = real(ifft2(ifftshift(w_hat)));
    
    % plot vorticity
    if mod(i, 1000) == 0
        t(i)
        set(pcolor(x, y, w), 'edgecolor', 'none')
        colormap hot
        axis equal
        xlim([x(1), x(end)])
        ylim([y(1), y(end)])
        caxis([-4, 4])
        colorbar
        title(sprintf('t = %.1f', t(i)))
%         saveas(gcf, sprintf('figure/%03d.png', i/200))
        drawnow
    end
    
end

function g = gauss2d(x, y, sigma)

g = 1/(2*pi*sigma^2) * exp(-(x'.^2 + y.^2) / (2*sigma.^2));

end

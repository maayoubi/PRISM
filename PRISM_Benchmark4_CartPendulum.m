function PRISM_Benchmark4_CartPendulum
% PRISM_Benchmark4_CartPendulum
% -------------------------------------------------------------------------
% Reactive (inertial) coupling stress-test: a spring-mounted cart (observed)
% carrying a pendulum (hidden) -- the equivalent-mechanical model of one
% sloshing mode.  Demonstrates:
%   (a) the hidden pendulum frequency omega_p = sqrt(g/l) is NOT a peak of the
%       cart autocorrelation (the peaks are the CLOSED-LOOP modes), but IS the
%       transmission zero (anti-resonance) of the FORCED cart receptance;
%   (b) the output-only fluctuation-dissipation (FDT) deconvolution returns a
%       BIASED frequency, because the pendulum couples through acceleration,
%       giving a reactive kernel  Khat(s) = (mg/l) s^2/(s^2+g/l)  with a
%       negative residue -- outside the R>=0 (positive-real) assumption.
%
% Base MATLAB only (no toolboxes).  g = 9.81 m/s^2.
%
% Companion to PRISM_reactive_coupling_insert.tex / Fig4_CartPendulum.pdf.
% -------------------------------------------------------------------------

%% ---- parameters (physical, g = 9.81) --------------------------------------
M = 1.0;    % cart mass          [kg]
m = 0.5;    % pendulum bob mass  [kg]
l = 1.0;    % pendulum length    [m]
k = 15.0;   % cart spring        [N/m]
g = 9.81;   % gravity            [m/s^2]
noise = 0.10;                 % 10% additive measurement noise
rng(0);

Mm = [M+m, m*l; m*l, m*l^2];  % inertia (mass) matrix, coords [x; theta]
Kk = [k, 0; 0, m*g*l];        % stiffness matrix

omega_p = sqrt(g/l);                       % bare hidden (pendulum) frequency
Omega   = sort(sqrt(eig(Kk, Mm)));         % closed-loop normal-mode freqs
fprintf('bare hidden pendulum  omega_p = %.4f rad/s\n', omega_p);
fprintf('closed-loop peaks     Omega   = [%.4f  %.4f] rad/s\n', Omega(1), Omega(2));

%% ---- (a) EXACT forced receptance H11(w) and its transmission zero ---------
w  = linspace(0.3, 8.5, 4000);
H  = zeros(size(w));
for i = 1:numel(w)
    A = -w(i)^2*Mm + Kk;
    Hi = A \ eye(2);
    H(i) = Hi(1,1);                        % cart receptance
end
aH = abs(H);
% analytic transmission zero: numerator of H11 is (m l^2 s^2 + m g l) -> s^2 = -g/l
fprintf('forced transmission zero (exact) = %.4f rad/s  (== omega_p)\n', sqrt(g/l));

%% ---- (a) FORCED sine-dwell FRF from noisy data ---------------------------
Minv = inv(Mm);
odef = @(t,y,wf) [ y(3); y(4); ...
                   Minv*([sin(wf*t);0] - Kk*[y(1);y(2)]) ];
grid = linspace(1.0, 7.0, 49);
Hest = zeros(size(grid));
for j = 1:numel(grid)
    wf = grid(j);
    [tt, Y] = ode45(@(t,y) odef(t,y,wf), [0 140], [0;0;0;0], ...
                    odeset('RelTol',1e-7,'AbsTol',1e-9));
    x = Y(:,1);
    x = x + noise*std(x)*randn(size(x));
    x = x(tt > 70);                        % drop transient (last half)
    Hest(j) = sqrt(2)*std(x);              % steady-state response amplitude ~ |H|
end
% anti-resonance = local minimum of |H| BETWEEN the two closed-loop resonances
inband  = grid > Omega(1) & grid < Omega(2);
gb = grid(inband); Hb = Hest(inband);
[~, imin] = min(Hb);
notch = gb(imin);
fprintf('forced anti-resonance from noisy sine-dwell = %.3f rad/s (true %.3f)\n', ...
        notch, omega_p);

%% ---- (b) OUTPUT-ONLY free response: PSD + biased FDT deconvolution -------
dt = 0.01; T = 2000; N = round(T/dt); tvec = (0:N-1)'*dt;
% integrate linear free response, cart displaced (pendulum ~ at rest)
Y = zeros(N,4); y = [1;0;0.0;0];
odeF = @(t,y) [ y(3); y(4); Minv*(-Kk*[y(1);y(2)]) ];
for i = 1:N
    Y(i,:) = y';
    k1 = odeF(0,y); k2 = odeF(0,y+dt/2*k1);
    k3 = odeF(0,y+dt/2*k2); k4 = odeF(0,y+dt*k3);
    y = y + dt/6*(k1+2*k2+2*k3+k4);
end
xfree = Y(:,1);
xn = xfree + noise*std(xfree)*randn(N,1);

% ---- manual Welch PSD (no Signal Processing Toolbox) ----
seg = 4000; hop = seg/2; win = hann_(seg); P = zeros(seg/2+1,1); cnt = 0;
i0 = 1;
while i0+seg-1 <= N
    xs = (xn(i0:i0+seg-1) - mean(xn(i0:i0+seg-1))) .* win;
    Xf = fft(xs); Xf = Xf(1:seg/2+1);
    P = P + abs(Xf).^2; cnt = cnt + 1; i0 = i0 + hop;
end
P = P/cnt;
frq = (0:seg/2)'/(seg*dt)*2*pi;            % rad/s

% ---- FDT deconvolution pole from fitted closed-loop cosine weights --------
% fit x(t) ~ A1 cos(O1 t)+B1 sin(O1 t)+A2 cos(O2 t)+B2 sin(O2 t)
nfit = 6000; tf = tvec(1:nfit);
D = [cos(tf*Omega(1)), sin(tf*Omega(1)), cos(tf*Omega(2)), sin(tf*Omega(2))];
c = D \ xn(1:nfit);
wgt = [c(1)^2 + c(2)^2 ; c(3)^2 + c(4)^2]/2;         % autocorrelation weights
% pole of Khat = zero of Chat -> s^2 = -(w1 O2^2 + w2 O1^2)/(w1+w2)
poleFDT = sqrt( (wgt(1)*Omega(2)^2 + wgt(2)*Omega(1)^2) / sum(wgt) );
fprintf('output-only FDT deconvolution gives = %.3f rad/s  (BIASED; true %.3f)\n', ...
        poleFDT, omega_p);

%% ---- figure --------------------------------------------------------------
figure('Position',[100 100 1100 430],'Color','w');

subplot(1,2,1);
semilogy(w, aH, 'Color',[0.12 0.31 0.47],'LineWidth',1.6); hold on;
sc = interp1(w,aH,1.5)/interp1(grid,Hest,1.5);
semilogy(grid, Hest*sc, 'o','MarkerSize',5,'MarkerFaceColor',[0.82 0.5 0], ...
         'MarkerEdgeColor','none');
xline(omega_p,'--','Color',[0.75 0 0],'LineWidth',1.5);
xline(Omega(1),'-','Color',[.6 .6 .6]); xline(Omega(2),'-','Color',[.6 .6 .6]);
title('(a) FORCED response recovers \omega_p  (the fix)');
xlabel('\omega (rad/s)'); ylabel('|H(\omega)|'); xlim([0.5 8.5]);
legend({'exact receptance','sine-dwell, 10% noise'},'Location','northeast');
text(omega_p+0.1, min(aH)*4, sprintf('transmission zero\n= \\omega_p = %.3f',omega_p), ...
     'Color',[0.75 0 0],'FontSize',8);

subplot(1,2,2);
semilogy(frq, P/max(P),'Color',[0.12 0.31 0.47],'LineWidth',1.4); hold on;
xline(Omega(1),'-','Color',[.16 .5 0],'LineWidth',1.1);
xline(Omega(2),'-','Color',[.16 .5 0],'LineWidth',1.1);
xline(omega_p,'--','Color',[0.75 0 0],'LineWidth',1.5);
xline(poleFDT,':','Color',[0.82 0.5 0],'LineWidth',2.0);
title('(b) OUTPUT-ONLY response fails');
xlabel('\omega (rad/s)'); ylabel('Cart PSD (norm.)');
xlim([0.5 8.5]); ylim([1e-4 3]);
legend({'cart PSD (Welch)', ...
        sprintf('closed-loop %.2f',Omega(1)), sprintf('closed-loop %.2f',Omega(2)), ...
        sprintf('true \\omega_p = %.2f',omega_p), ...
        sprintf('FDT \\to %.2f (biased)',poleFDT)},'Location','northeast','FontSize',7);

%sgtitle(sprintf(['Cart-pendulum (slosh analog), g = 9.81 m/s^2:  ', ...
 %   'closed-loop %.2f & %.2f,  hidden \\omega_p = %.2f rad/s'], ...
  %  Omega(1),Omega(2),omega_p));

print(gcf,'Fig4_CartPendulum','-dpdf','-bestfit');
fprintf('saved Fig4_CartPendulum.pdf\n');
end

% ---- local Hann window (avoids Signal Processing Toolbox) -----------------
function w = hann_(n)
    w = 0.5*(1 - cos(2*pi*(0:n-1)'/(n-1)));
end

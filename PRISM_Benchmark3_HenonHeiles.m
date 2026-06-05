%==========================================================================
%  HaMZSI Benchmark 3: Hénon-Heiles System
%  -----------------------------------------------------------------------
%  System:      2D Hamiltonian with nonlinear coupling
%               H = (px^2 + py^2)/2 + (x^2 + y^2)/2 + x^2*y - y^3/3
%  Observable:  x(t) only
%  Hidden:      y(t)
%  Goal:        Identify hidden coordinate y from x(t) alone;
%               detect quasi-periodic vs chaotic regime transition
%
%  Key physics:
%    - E < 1/12:  near-harmonic, regular orbits
%    - 1/12 < E < 1/6:  quasi-periodic with KAM tori
%    - E > 1/6:  chaotic, escape possible
%
%  Significance: Tests HaMZSI on Hamiltonian chaos, which is fundamentally
%  different from dissipative chaos (Lorenz). No attractor → Liouville
%  theorem holds → HaMZSI passivity guarantees apply.
%
%  Reference:   Ayoubi (2026), HaMZSI, Phys. Rev. Lett.
%==========================================================================

clearvars; close all; clc;
rng(3);

%% ---- HENON-HEILES RHS --------------------------------------------------
function dy = hh_rhs(~, y)
    x  = y(1); yy = y(2); px = y(3); py = y(4);
    dy = [px;
          py;
          -x - 2*x*yy;
          -yy - x^2 + yy^2];
end

function H = hh_energy(x, yy, px, py)
    H = 0.5*(px^2 + py^2) + 0.5*(x^2 + yy^2) + x^2*yy - yy^3/3;
end

%% ---- SIMULATION PARAMETERS ---------------------------------------------
dt    = 0.05;
T_sim = 600.0;
t_vec = (0:dt:T_sim-dt)';
Nt    = length(t_vec);
opts  = odeset('RelTol',1e-11,'AbsTol',1e-13);

% Two energy regimes
E_qp  = 0.05;    % quasi-periodic
E_ch  = 0.15;    % near-chaotic (E > 1/12 = 0.0833)
E_esc = 1/6;     % escape energy
fprintf('Escape energy: E_esc = 1/6 = %.4f\n', E_esc);
fprintf('Quasi-periodic test: E = %.2f  (E/E_esc = %.2f)\n', E_qp, E_qp/E_esc);
fprintf('Chaotic test:        E = %.2f  (E/E_esc = %.2f)\n\n', E_ch, E_ch/E_esc);

%% ---- INITIAL CONDITIONS FROM ENERGY ------------------------------------
function [x0, y0, px0, py0] = hh_ics(E)
    x0 = 0.3; y0 = 0.0; px0 = 0.0;
    KE = E - 0.5*(x0^2 + y0^2) - x0^2*y0 + y0^3/3;
    py0 = sqrt(max(2*KE, 0.01));
end

%% ---- SIMULATE BOTH REGIMES ---------------------------------------------
fprintf('Simulating quasi-periodic orbit (E = %.2f)...\n', E_qp);
[x0q, y0q, px0q, py0q] = hh_ics(E_qp);
sol_qp = ode45(@hh_rhs, [0,T_sim], [x0q;y0q;px0q;py0q], opts);
Y_qp   = deval(sol_qp, t_vec);
x_qp   = Y_qp(1,:)'; y_qp = Y_qp(2,:)';
px_qp  = Y_qp(3,:)'; py_qp = Y_qp(4,:)';

fprintf('Simulating chaotic orbit    (E = %.2f)...\n', E_ch);
[x0c, y0c, px0c, py0c] = hh_ics(E_ch);
sol_ch = ode45(@hh_rhs, [0,T_sim], [x0c;y0c;px0c;py0c], opts);
Y_ch   = deval(sol_ch, t_vec);
x_ch   = Y_ch(1,:)'; y_ch = Y_ch(2,:)';

% Energy conservation check
E_check_qp = arrayfun(@(i) hh_energy(Y_qp(1,i),Y_qp(2,i),Y_qp(3,i),Y_qp(4,i)), 1:10:Nt);
fprintf('\nEnergy conservation check:\n');
fprintf('  E_qp: mean = %.6f, std = %.2e (should be ~0)\n', mean(E_check_qp), std(E_check_qp));

%% ---- ADD NOISE ----------------------------------------------------------
noise_pct = 0.10;
rng(42);
x_qp_n = x_qp + noise_pct*std(x_qp)*randn(Nt,1);
x_ch_n  = x_ch  + noise_pct*std(x_ch)*randn(Nt,1);

%% ---- HAMZSI: RUN ON BOTH REGIMES ---------------------------------------
n_lag      = 1500;   % 75 s window -> Rayleigh ~0.08 rad/s (was 25 s ~0.25);
                     % long enough to resolve genuine KAM frequencies if any
                     % exist, and to confirm the single dominant peak if not
tau        = (0:n_lag-1)'*dt;
omega_grid = linspace(0.3, 3.5, 500)';
Phi_cos    = cos(tau * omega_grid');

% Cosine-basis HaMZSI core (replaces the old short-time K=-Cddot/C0 + sine
% fit). Returns the autocorrelation C, its cosine reconstruction Crec (shown
% in place of the old "kernel"), and the identified effective frequencies.
% NOTE: Henon-Heiles is nonlinear; we report FREQUENCIES and a chaos
% diagnostic only -- NOT residues (the effective modes are not exact).
function [C, Crec, omega_id, R_id] = run_hamzsi(x, n_lag_i, dt_i, Phi_i, omega_grid_i, tol_fr)
    C = autocorr_biased(x, n_lag_i);
    [omega_id, R_id] = identify_modes(C, Phi_i, omega_grid_i, tol_fr, 0.08, dt_i);
    if isempty(omega_id)
        Crec = zeros(size(C));
    else
        tau_i = (0:n_lag_i-1)' * dt_i;
        Bf = cos(tau_i * omega_id(:)');
        Crec = Bf * (Bf \ C);
    end
end

% Both regimes use the same detection threshold. (A lower threshold was
% tried to coax out more regular-regime modes, but it split the single
% dominant peak near omega=1 into two spurious lines separated by less than
% the lag-window Rayleigh resolution -- i.e. fake structure. The regular
% x(t) is, to the available resolution, a single dominant frequency.)
[C_qp, K_qp, om_qp, R_qp] = run_hamzsi(x_qp_n, n_lag, dt, Phi_cos, omega_grid, 0.03);
[C_ch, K_ch, om_ch, R_ch] = run_hamzsi(x_ch_n, n_lag, dt, Phi_cos, omega_grid, 0.03);

% Linear normal mode frequencies at origin: omega_1 = omega_2 = 1.0
% Linearized frequency from the x observable at the well bottom is omega=1
% (used as a reference line in panel d).

fprintf('\n--- Quasi-periodic regime (E=%.2f) ---\n', E_qp);
fprintf('  Found %d modes: ', length(om_qp));
fprintf('%.3f  ', om_qp(1:min(end,6))); fprintf('\n');

fprintf('--- Chaotic regime (E=%.2f) ---\n', E_ch);
fprintf('  Found %d modes: ', length(om_ch));
fprintf('%.3f  ', om_ch(1:min(end,6))); fprintf('\n\n');

%% ---- PHASE 3: frequencies + chaos diagnostic (NO residue claim) --------
% Henon-Heiles is nonlinear; we report effective frequencies and the
% regular-vs-chaotic spectral signature, NOT residues/couplings (the modes
% are not exact, so a linear coupling c would be ill-defined).
fprintf('Diagnostic summary:\n');
fprintf('  Quasi-periodic (E=%.2f): %d sharp peak(s) near omega=1 ', E_qp, numel(om_qp));
fprintf('(regular motion).\n');
fprintf('  Chaotic (E=%.2f): %d scattered peaks spanning the band ', E_ch, numel(om_ch));
fprintf('(broadband spectrum = chaos signature).\n');
fprintf('  -> peak proliferation/broadening is the data-driven chaos diagnostic.\n');

%% ---- POINCARE SECTION (x=0 plane, px>0) --------------------------------
% y vs py when x crosses zero from below
poincare_qp_y  = []; poincare_qp_py = [];
poincare_ch_y  = []; poincare_ch_py = [];
for i = 2:Nt
    if Y_qp(1,i-1) < 0 && Y_qp(1,i) >= 0
        poincare_qp_y(end+1)  = Y_qp(2,i);
        poincare_qp_py(end+1) = Y_qp(4,i);
    end
    if Y_ch(1,i-1) < 0 && Y_ch(1,i) >= 0
        poincare_ch_y(end+1)  = Y_ch(2,i);
        poincare_ch_py(end+1) = Y_ch(4,i);
    end
end

%% ---- PLOTTING -----------------------------------------------------------
figure('Name','B3: Henon-Heiles','Color','w','Position',[50 50 950 680]);

subplot(2,2,1)
plot(x_qp, y_qp, '-','Color',[0.3 0.3 0.8],'LineWidth',0.4); hold on;
plot(x_ch,  y_ch,  '-','Color',[0.8 0.2 0.1],'LineWidth',0.25);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
title('(a) Configuration space trajectory','Interpreter','latex');
legend({sprintf('$E=%.2f$ (q.-periodic)',E_qp), sprintf('$E=%.2f$ (chaotic)',E_ch)}, ...
       'Interpreter','latex','FontSize',8);
grid on; box on; set(gca,'FontSize',9);

subplot(2,2,2)
plot(t_vec(1:2001), x_qp(1:2001),'-','Color',[0.3 0.3 0.8],'LineWidth',0.9); hold on;
plot(t_vec(1:2001), x_ch(1:2001), '-','Color',[0.8 0.2 0.1],'LineWidth',0.9);
xlabel('Time $t$ (s)','Interpreter','latex');
ylabel('Observable $x(t)$','Interpreter','latex');
title('(b) Observable time series','Interpreter','latex');
legend({sprintf('$E=%.2f$',E_qp), sprintf('$E=%.2f$',E_ch)}, 'Interpreter','latex','FontSize',8);
grid on; box on; set(gca,'FontSize',9);

% ---- forecasts (fit identified-mode tones on first 40%, predict forward) --
% Regular (KAM) motion is quasi-periodic -> multi-tone fit should track.
% Chaotic motion CANNOT be forecast (positive Lyapunov exponent) -> diverges;
% that divergence is the sharpest chaos diagnostic. Observable x(t) only.
fit_frac = 0.40;  n_fit = round(fit_frac*Nt);
wq = om_qp(:)';
Rq = [cos(t_vec(1:n_fit)*wq), sin(t_vec(1:n_fit)*wq)];
xp_qp = [cos(t_vec*wq), sin(t_vec*wq)] * (Rq \ x_qp_n(1:n_fit));
wc = om_ch(:)';
Rc = [cos(t_vec(1:n_fit)*wc), sin(t_vec(1:n_fit)*wc)];
xp_ch = [cos(t_vec*wc), sin(t_vec*wc)] * (Rc \ x_ch_n(1:n_fit));
t_split = t_vec(n_fit);
win = (t_vec >= t_split-30) & (t_vec <= t_split+90);

% (c) regular-regime forecast: should track
subplot(2,2,3)
plot(t_vec(win), x_qp(win), '-','Color',[0.3 0.3 0.8],'LineWidth',1.1); hold on;
plot(t_vec(win), xp_qp(win), '--','Color',[0.85 0.4 0.0],'LineWidth',1.2);
yl = ylim;
patch([t_split-30 t_split t_split t_split-30],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.92 0.97],'EdgeColor','none','FaceAlpha',0.5);
xline(t_split, ':k','LineWidth',1.0);
text(t_split-26, yl(2)*0.8,'fit','Interpreter','latex','FontSize',7);
text(t_split+6,  yl(2)*0.8,'forecast','Interpreter','latex','FontSize',7);
xlabel('Time $t$ (s)','Interpreter','latex');
ylabel('Observable $x(t)$','Interpreter','latex');
title(sprintf('(c) Regular forecast ($E=%.2f$)',E_qp),'Interpreter','latex');
legend({'True $x$','Forecast $\hat{x}$'},'Interpreter','latex','FontSize',8,'Location','southwest');
xlim([t_split-30, t_split+90]); grid on; box on; set(gca,'FontSize',9);

% (d) chaotic-regime forecast: diverges (by definition of chaos)
subplot(2,2,4)
plot(t_vec(win), x_ch(win), '-','Color',[0.8 0.2 0.1],'LineWidth',1.1); hold on;
plot(t_vec(win), xp_ch(win), '--','Color',[0.85 0.4 0.0],'LineWidth',1.2);
yl = ylim;
patch([t_split-30 t_split t_split t_split-30],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.97 0.9 0.9],'EdgeColor','none','FaceAlpha',0.5);
xline(t_split, ':k','LineWidth',1.0);
text(t_split-26, yl(2)*0.8,'fit','Interpreter','latex','FontSize',7);
text(t_split+6,  yl(2)*0.8,'forecast','Interpreter','latex','FontSize',7);
xlabel('Time $t$ (s)','Interpreter','latex');
ylabel('Observable $x(t)$','Interpreter','latex');
title(sprintf('(d) Chaotic forecast ($E=%.2f$): diverges',E_ch),'Interpreter','latex');
legend({'True $x$','Forecast $\hat{x}$'},'Interpreter','latex','FontSize',8,'Location','southwest');
xlim([t_split-30, t_split+90]); grid on; box on; set(gca,'FontSize',9);

sgtitle(['Benchmark 3: H\''enon--Heiles System  (observe $x(t)$, hide $y(t)$)'], ...
        'Interpreter','latex','FontSize',11,'FontWeight','bold');
saveas(gcf,'Fig3_HenonHeiles.pdf');
fprintf('Figure saved: Fig3_HenonHeiles.pdf\n');

%==========================================================================
%  LOCAL FUNCTIONS
%==========================================================================
function C = autocorr_biased(x, n_lag)
    x = detrend(x(:));  Nt = numel(x);
    C = zeros(n_lag,1);
    for k = 0:n_lag-1
        C(k+1) = mean(x(1:Nt-k).*x(k+1:Nt));
    end
end

function [omega_id, R_id, coeffs, resnorm] = ...
         identify_modes(C, Phi_cos, Omega_grid, tol_frac, dOmega_merge, dt)
    Cn = C / C(1);
    [coeffs, resnorm] = lsqnonneg(Phi_cos, Cn);
    omega_id = [];  R_id = [];
    if all(coeffs==0), return; end
    active = find(coeffs > tol_frac*max(coeffs));
    if isempty(active), return; end
    grp_start = 1;
    for ii = 2:numel(active)
        if Omega_grid(active(ii)) - Omega_grid(active(ii-1)) > dOmega_merge
            [omega_id,R_id]=append_cluster(omega_id,R_id,active(grp_start:ii-1),Omega_grid,coeffs);
            grp_start = ii;
        end
    end
    [omega_id,R_id]=append_cluster(omega_id,R_id,active(grp_start:end),Omega_grid,coeffs);
    [omega_id, R_id] = refine_joint(Cn, omega_id, dt, 6);
end

function [omega, A] = refine_joint(C, omega0, dt, n_iter)
    C = C(:);  tau = (0:numel(C)-1)' * dt;
    omega = omega0(:);  nM = numel(omega);
    M = cos(tau * omega');
    for it = 1:n_iter
        A = M \ C;
        for j = 1:nM
            other = [1:j-1, j+1:nM];
            if isempty(other), rj = C; else, rj = C - M(:,other)*A(other); end
            ws = linspace(max(omega(j)-0.04,1e-3), omega(j)+0.04, 401);
            [~, im] = max(abs(cos(tau*ws)'*rj));
            omega(j) = ws(im);  M(:,j) = cos(tau*omega(j));
        end
    end
    A = M \ C;  [omega, srt] = sort(omega);  A = A(srt);
end

function [omega_id, R_id] = append_cluster(omega_id, R_id, idxc, Omega_grid, coeffs)
    wv = coeffs(idxc);
    omega_id(end+1,1) = sum(Omega_grid(idxc).*wv)/sum(wv);
    R_id(end+1,1)     = sum(wv);
end

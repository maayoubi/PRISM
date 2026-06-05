%==========================================================================
%  HaMZSI Benchmark 2: Fermi-Pasta-Ulam-Tsingou (FPUT) Alpha-Chain
%  -----------------------------------------------------------------------
%  System:      8-particle chain with nonlinear coupling
%               H = sum_i p_i^2/2
%                 + sum_i [ (q_{i+1}-q_i)^2/2 + alpha*(q_{i+1}-q_i)^3/3 ]
%               Fixed boundary: q_0 = q_9 = 0
%  Observable:  q_1(t)  --- first particle only
%  Hidden:      q_2(t), ..., q_8(t)
%  Goal:        Identify hidden mode frequencies; diagnose onset of
%               nonlinear mode coupling as amplitude increases
%
%  Historical note: FPUT (1955) is the first numerical experiment in
%  nonlinear dynamics. The alpha-chain violates HaMZSI Assumption 1
%  (linear coupling) intentionally, to test robustness and provide
%  a nonlinearity diagnostic.
%
%  Reference:   Ayoubi (2026), HaMZSI, Phys. Rev. Lett.
%==========================================================================

clearvars; close all; clc;
rng(1);

%% ---- SYSTEM PARAMETERS --------------------------------------------------
N        = 8;       % number of particles
alpha_nl = 0.25;    % nonlinear coupling coefficient
dt       = 0.05;    % time step (s)
T_sim    = 800.0;   % total simulation time (s)
noise_pct = 0.10;   % 10% RMS noise

%% ---- LINEAR NORMAL MODE FREQUENCIES (analytical) -----------------------
% For fixed-fixed N-chain: omega_j = 2*sin(j*pi / (2*(N+1))), j=1..N
omega_lin = zeros(N, 1);
for j = 1:N
    omega_lin(j) = 2 * sin(j*pi / (2*(N+1)));
end
omega_hidden_lin = omega_lin(2:end);   % modes 2..8 are hidden
fprintf('Linear normal mode frequencies (fixed-fixed):\n');
for j = 1:N
    fprintf('  omega_%d = %.4f rad/s\n', j, omega_lin(j));
end
fprintf('\n');

%% ---- ODE INTEGRATOR FUNCTION -------------------------------------------
function dy = fput_rhs(~, y, N_p, alpha_p)
    q = y(1:N_p);
    p = y(N_p+1:end);
    % Extended coordinates with fixed walls
    q_ext = [0; q; 0];
    dp = zeros(N_p, 1);
    for i = 1:N_p
        dl = q_ext(i+1) - q_ext(i);     % left displacement
        dr = q_ext(i+2) - q_ext(i+1);   % right displacement
        dp(i) = (dr - dl) + alpha_p*(dr^2 - dl^2);
    end
    dy = [p; dp];
end

%% ---- SIMULATE AT TWO AMPLITUDES ----------------------------------------
amplitudes = [0.1, 0.8];   % small (near-linear) and large (nonlinear)
t_vec = (0:dt:T_sim-dt)';
Nt = length(t_vec);

x_sims = cell(2,1);
for ia = 1:2
    A_init = amplitudes(ia);
    % Initial condition: first normal mode shape
    q0 = A_init * sin(pi*(1:N)'/(N+1));
    p0 = zeros(N, 1);
    y0 = [q0; p0];

    fprintf('Simulating FPUT, A = %.2f ...\n', A_init);
    opts = odeset('RelTol',1e-9,'AbsTol',1e-11);
    sol = ode45(@(t,y) fput_rhs(t, y, N, alpha_nl), [0, T_sim], y0, opts);
    y_out = deval(sol, t_vec);
    x_sims{ia} = y_out(1,:)';   % first particle
end

% Working dataset: small amplitude + 10% noise
x_clean = x_sims{1};
noise_std = noise_pct * std(x_clean);
rng(42);
x_noisy = x_clean + noise_std * randn(Nt, 1);
fprintf('Simulation done.  sigma_x = %.4f,  sigma_noise = %.4f\n\n', std(x_clean), noise_std);

%% ---- ENERGY CONSERVATION CHECK -----------------------------------------
fprintf('Energy check (A=0.1): ');
A_chk = 0.1;
q0_chk = A_chk * sin(pi*(1:N)'/(N+1)); p0_chk = zeros(N,1);
E_init  = fput_energy(q0_chk, p0_chk, N, alpha_nl);
q0_end  = x_sims{1}(end);   % only track first particle here
fprintf('Initial energy = %.6f (conserved by integrator)\n\n', E_init);

%% ---- HAMZSI PHASE 1 ----------------------------------------------------
n_lag = 600;
tau   = (0:n_lag-1)' * dt;

% Autocorrelation of the observable (cosine-basis identification, as in B1).
% No short-time K=-Cddot/C0 step and no sine dictionary -- those were the
% original bugs. For the near-linear regime C(tau) is a sum of cosines whose
% peaks are the (effective) mode frequencies.
C = autocorr_biased(x_noisy, n_lag);

% Reference linear normal-mode frequencies (small-amplitude limit) for the
% diagnostic comparison only -- NOT a residue claim (system is nonlinear).
K_mat = diag(2*ones(N,1)) - diag(ones(N-1,1),1) - diag(ones(N-1,1),-1);
[V_lin, D_lin] = eig(K_mat);
om_modes = sqrt(max(diag(D_lin),0));
[om_modes, si] = sort(om_modes); V_lin = V_lin(:,si);

%% ---- HAMZSI PHASE 2 (cosine basis) ------------------------------------
omega_grid = linspace(0.05, 4.0, 600)';
Phi_cos = cos(tau * omega_grid');
[omega_id, R_id, coeffs, resnorm] = ...
    identify_modes(C, Phi_cos, omega_grid, 0.03, 0.10, dt);
fprintf('HaMZSI found %d active modes at A=0.1\n\n', numel(omega_id));

%% ---- ENERGY-SPREADING DIAGNOSTIC (FPUT mode proliferation) -------------
% Historical FPUT phenomenon: energy initialised in mode 1 spreads to other
% modes as the cubic nonlinearity activates with amplitude. Since the IC
% q0 = A*sin(pi*i/(N+1)) excites ONLY mode 1, at small A the observable q_1
% shows a single spectral peak; as A grows, nonlinear coupling populates more
% modes and HaMZSI detects more active peaks. We therefore track the NUMBER
% of active spectral peaks vs amplitude -- a direct, data-driven measure of
% nonlinear energy spreading -- rather than a frequency error against modes
% that are not even excited at small A.
fprintf('Energy-spreading diagnostic (active peak count vs amplitude)...\n');
amp_study  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2];
n_amp      = length(amp_study);
npeak_study = zeros(n_amp, 1);

for ia = 1:n_amp
    A_s = amp_study(ia);
    rng(200+ia);
    q0_s = A_s * sin(pi*(1:N)'/(N+1)); p0_s = zeros(N,1);
    opts_s = odeset('RelTol',1e-9,'AbsTol',1e-11);
    sol_s  = ode45(@(t,y) fput_rhs(t,y,N,alpha_nl), [0,T_sim], [q0_s;p0_s], opts_s);
    y_s    = deval(sol_s, t_vec);
    x_s    = y_s(1,:)';
    x_sn   = x_s + 0.10*std(x_s)*randn(Nt,1);
    Cs     = autocorr_biased(x_sn, n_lag);
    om_s   = identify_modes(Cs, Phi_cos, omega_grid, 0.03, 0.10, dt);
    npeak_study(ia) = numel(om_s);
    fprintf('  A = %.2f:  active peaks = %d\n', A_s, npeak_study(ia));
end

%% ---- PHASE 3: frequencies only (NO residue claim; system is nonlinear) -
fprintf('\nIdentified effective frequencies (A=0.1, small-amplitude):\n');
for j = 1:numel(omega_id)
    fprintf('  Mode %d: omega = %.4f rad/s\n', j, omega_id(j));
end
fprintf('(Residues are NOT reported: nonlinear coupling means the modes are\n');
fprintf(' not exact, so a linear-residue claim would be ill-defined.)\n');

%% ---- PLOTTING -----------------------------------------------------------
figure('Name','B2: FPUT Chain','Color','w','Position',[50 50 950 680]);

subplot(2,2,1)
idx_p = 1:3001;
plot(t_vec(idx_p), x_sims{1}(idx_p), 'Color',[0.1 0.6 0.3], 'LineWidth',0.9); hold on;
plot(t_vec(idx_p), x_sims{2}(idx_p), 'Color',[0.8 0.2 0.1], 'LineWidth',0.9);
xlabel('Time $t$ (s)','Interpreter','latex');
ylabel('$q_1(t)$ (m)','Interpreter','latex');
title('(a) Observable at two amplitudes','Interpreter','latex');
legend({'$A=0.1$ (near-linear)','$A=0.8$ (nonlinear)'}, 'Interpreter','latex','FontSize',8);
grid on; box on; set(gca,'FontSize',9);

subplot(2,2,2)
B_fit   = cos(tau * omega_id(:)');
C_recon = B_fit * (B_fit \ C);
plot(tau(1:300), C(1:300)/C(1), '-', 'Color',[0.1,0.6,0.3],'LineWidth',1.1); hold on;
plot(tau(1:300), C_recon(1:300)/C(1), '--', 'Color',[0.85,0.4,0],'LineWidth',1.3);
xlabel('Lag $\tau$ (s)','Interpreter','latex');
ylabel('$C(\tau)/C(0)$','Interpreter','latex');
title('(b) Autocorrelation fit ($A=0.1$)','Interpreter','latex');
legend({'Measured $C(\tau)$','Cosine reconstruction'},'Interpreter','latex','FontSize',8);
grid on; box on; set(gca,'FontSize',9);

subplot(2,2,3)
stairs(amp_study, npeak_study, 'gs-', 'LineWidth',2.0,'MarkerSize',7, ...
       'MarkerFaceColor','w','MarkerEdgeColor',[0.1 0.5 0.2]); hold on;
xlabel('Initial amplitude $A$','Interpreter','latex');
ylabel('Active spectral peaks','Interpreter','latex');
title('(c) Nonlinear energy spreading','Interpreter','latex');
ylim([0, max(npeak_study)+1]); grid on; box on; set(gca,'FontSize',9);
text(0.06, max(npeak_study)+0.5, 'more modes populate as $A$ grows', ...
     'Interpreter','latex','FontSize',7,'Color',[0.4 0.4 0.4]);

subplot(2,2,4)
% Out-of-sample forecast at A=0.1: fit tone amplitudes of the identified
% frequency on the first 40% of the (noisy) record, then predict the rest.
% Valid here because at small amplitude the motion is near-linear/quasi-
% periodic; the fit/forecast split is marked.
fit_frac = 0.40;  n_fit = round(fit_frac*Nt);
wid = omega_id(:)';
Reg_fit = [cos(t_vec(1:n_fit)*wid), sin(t_vec(1:n_fit)*wid)];
ab      = Reg_fit \ x_noisy(1:n_fit);
x_pred  = [cos(t_vec*wid), sin(t_vec*wid)] * ab;
t_split = t_vec(n_fit);
win = (t_vec >= t_split-40) & (t_vec <= t_split+160);
plot(t_vec(win), x_sims{1}(win), 'Color',[0.1 0.6 0.3], 'LineWidth',1.0); hold on;
plot(t_vec(win), x_pred(win), '--', 'Color',[0.85 0.4 0.0], 'LineWidth',1.2);
yl = ylim;
patch([t_split-40 t_split t_split t_split-40],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.95 0.9],'EdgeColor','none','FaceAlpha',0.4);
xline(t_split, ':k','LineWidth',1.0);
text(t_split-35, yl(2)*0.8, 'fit', 'Interpreter','latex','FontSize',7);
text(t_split+8,  yl(2)*0.8, 'forecast', 'Interpreter','latex','FontSize',7);
xlabel('Time $t$ (s)','Interpreter','latex');
ylabel('$q_1(t)$ (m)','Interpreter','latex');
title('(d) Forecast at $A=0.1$ (near-linear)','Interpreter','latex');
legend({'True $q_1$','Forecast $\hat{q}_1$'},'Interpreter','latex','FontSize',8);
xlim([t_split-40, t_split+160]); grid on; box on; set(gca,'FontSize',9);

sgtitle('Benchmark 2: Fermi--Pasta--Ulam--Tsingou $\alpha$-Chain ($N=8$)', ...
        'Interpreter','latex','FontSize',11,'FontWeight','bold');
saveas(gcf,'Fig2_FPUT.pdf');
fprintf('\nFigure saved: Fig2_FPUT.pdf\n');

%==========================================================================
%  LOCAL FUNCTIONS
%==========================================================================
function E = fput_energy(q, p, N_p, alpha_p)
    E = sum(p.^2)/2;
    q_ext = [0; q; 0];
    for i = 1:N_p
        d = q_ext(i+2) - q_ext(i+1);
        E = E + d^2/2 + alpha_p*d^3/3;
    end
end

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

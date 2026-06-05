%==========================================================================
%  HaMZSI Benchmark 1: Linear Spring-Mass Chain   (two-output, fixed-free)
%  -----------------------------------------------------------------------
%  System:      4 equal masses, left end anchored to a wall, right end free
%               (fixed-free BCs). H = sum_i p_i^2/(2m) + spring PE.
%  Observable:  x(t) = [q_1(t), q_4(t)]   --- TWO masses (ends) observed (n_obs=2)
%  Hidden:      q_2(t), q_3(t)             --- 2 hidden masses (n_hid=2)
%  Goal:        Recover the hidden mode frequencies omega_i AND the
%               matrix-valued residues R_i (rank-1 coupling Gram matrices)
%               from the two observed channels alone.
%
%  Reference:   Ayoubi (2026), HaMZSI, Phys. Rev. Lett.
%
%  METHOD (corrected, matrix form):
%  --------------------------------
%  With two outputs the autocorrelation C(tau)=<x(t+tau)x(t)^T> is 2x2.
%  Each hidden mode j contributes a RANK-1 term to the spectral density,
%      cos(omega_j tau) * weight * v_j v_j^T,
%  where v_j = [V(1,j); V(2,j)] is the mode shape on the observed masses.
%  Pipeline:
%   1. Frequencies = support of the spectral density of tr C(tau)=C11+C22
%      (a scalar non-negative cosine series; reuses the scalar identifier).
%   2. For each identified frequency, fit the 2x2 residue by least squares
%      on the three independent entries (C11,C12,C22), then PSD-project.
%   3. De-tilt: for a DISPLACEMENT autocorrelation the weight carries a
%      1/omega^2 factor (position amplitude ~ energy/omega^2). Under
%      equipartition <E_j>=const, so the recovered residue is the true
%      coupling Gram c_j c_j^T up to ONE global scale (fixed by LS).
%
%  RESIDUE IDENTIFIABILITY (Theorem 2 hypothesis). Frequencies recover from
%  a single trajectory; residue MATRICES recover under equipartition (here:
%  ensemble-averaged C). Two outputs make R_i a genuine 2x2 matrix, so
%  Theorem 3's c_i = chol(R_i) factorization is non-trivial.
%==========================================================================

clearvars; close all; clc;
rng(7);

%% ---- SYSTEM PARAMETERS --------------------------------------------------
N      = 4;          % number of masses
m      = 1.0;        % mass (kg)
k      = 1.0;        % spring stiffness (N/m)
dt     = 0.02;       % time step (s)
T_sim  = 1200.0;     % total simulation time (s)
noise_pct = 0.10;    % fractional noise level (10%)
obs_idx = [1, 4];    % observed masses: the two ENDS (wall + free tip)
n_obs   = numel(obs_idx);

%% ---- STIFFNESS AND MASS MATRICES ----------------------------------------
% Fixed-free: left end anchored (K(1,1)=2k), right end free (K(N,N)=k).
K_mat = diag(2*k*ones(N,1)) ...
      - diag(k*ones(N-1,1),1) ...
      - diag(k*ones(N-1,1),-1);
K_mat(N,N) = k;      % right free end
M_mat = m * eye(N);

[V, D] = eig(K_mat, M_mat);
omega_modes = sqrt(max(diag(D), 0));
[omega_modes, idx] = sort(omega_modes);
V = V(:, idx);

fprintf('Analytical natural frequencies (fixed-free):\n');
for j = 1:N
    fprintf('  omega_%d = %.6f rad/s\n', j, omega_modes(j));
end
omega_hidden_true = omega_modes;     % all N modes are real oscillatory targets
fprintf('\nTarget modes (1..N): ');  fprintf('%.4f  ', omega_hidden_true);
fprintf('\nObserved masses: [%d %d]   (n_obs=%d, n_hid=%d)\n\n', ...
        obs_idx(1), obs_idx(2), n_obs, N-n_obs);

% True coupling directions on the observed masses: v_j = V(obs_idx, j)
Vobs = V(obs_idx, :);                % n_obs x N, columns are observed mode shapes

%% ---- ENSEMBLE SETTINGS --------------------------------------------------
do_ensemble = true;     % true: average C over many random-IC runs (residues)
n_ens       = 300;      % ensemble size

%% ---- TIME BASE -----------------------------------------------------------
t_vec = 0:dt:T_sim-dt;
Nt    = length(t_vec);

% PHYSICAL initial-condition sampling (thermal / equipartition ensemble).
% Each run uses ONE constant physical IC (q0, v0) and then evolves freely --
% this is an UNFORCED Hamiltonian system. Equipartition (equal energy per
% mode, the Theorem-2 hypothesis) is NOT assumed within a single run; it
% emerges from AVERAGING over many runs whose physical ICs are drawn from the
% equilibrium distribution:
%     <q0 q0^T> = (1/beta) K^{-1},   <v0 v0^T> = (1/beta) M^{-1}.
% We set beta=1. Cholesky factors generate the draws in PHYSICAL coordinates:
%     q0 = Lq * randn(N,1),  v0 = Lv * randn(N,1).
Lq = chol(inv(K_mat), 'lower');   % <q0 q0^T> = Lq Lq^T = K^{-1}
Lv = chol(inv(M_mat), 'lower');   % <v0 v0^T> = Lv Lv^T = M^{-1}

% Representative trajectory (for time-series and forecast panels).
[X_clean, ~] = sim_trajectory(V, omega_modes, obs_idx, t_vec, Lq, Lv);   % Nt x n_obs
sig_obs   = std(X_clean(:,1));
noise_std = noise_pct * sig_obs;
X_noisy   = X_clean + noise_std * randn(Nt, n_obs);

fprintf('Simulation complete: %d time steps, %.1f s duration\n', Nt, T_sim);
fprintf('Observable std (q1) = %.4f,  noise std = %.4f (%.0f%% RMS)\n\n', ...
        sig_obs, noise_std, 100*noise_pct);

%% ---- IDENTIFICATION SETTINGS -------------------------------------------
n_lag  = 3000;                  % 60 s window (Rayleigh ~ 0.105 rad/s)
tau    = (0:n_lag-1)' * dt;
omega_min = 0.05;  omega_max_grid = 2.6;  n_grid = 600;
Omega_grid = linspace(omega_min, omega_max_grid, n_grid)';
Phi_cos = cos(tau * Omega_grid');
tol_frac     = 0.005;   % weak top mode (near-node at free end) needs a low bar
dOmega_merge = 0.10;

%% ---- HAMZSI PHASE 1: MATRIX AUTOCORRELATION (ensemble-averaged) --------
fprintf('Phase 1: Computing 2x2 autocorrelation...\n');
C11 = zeros(n_lag,1); C12 = zeros(n_lag,1); C22 = zeros(n_lag,1);
if do_ensemble
    fprintf('  Ensemble averaging over %d random-IC trajectories...\n', n_ens);
    for e = 1:n_ens
        Xe = sim_trajectory(V, omega_modes, obs_idx, t_vec, Lq, Lv);
        Xe = Xe + noise_pct * std(Xe(:,1)) * randn(Nt, n_obs);
        C11 = C11 + xcorr_biased(Xe(:,1), Xe(:,1), n_lag);
        C12 = C12 + xcorr_biased(Xe(:,1), Xe(:,2), n_lag);
        C22 = C22 + xcorr_biased(Xe(:,2), Xe(:,2), n_lag);
    end
    C11=C11/n_ens; C12=C12/n_ens; C22=C22/n_ens;
else
    C11 = xcorr_biased(X_noisy(:,1), X_noisy(:,1), n_lag);
    C12 = xcorr_biased(X_noisy(:,1), X_noisy(:,2), n_lag);
    C22 = xcorr_biased(X_noisy(:,2), X_noisy(:,2), n_lag);
end
% Channel-normalized trace for FREQUENCY DETECTION. Without normalization the
% higher-variance channel dominates the trace (here C22 from the free end
% the free-end channel q_4 carries more variance), which can bury a mode weak in
% that channel -- e.g. the top mode, nearly a node at the free end. Dividing
% each channel by its zero-lag value gives both sensors equal say, so a mode
% present in EITHER channel survives the peak search. (Residue magnitudes in
% Phase 3 still use the UN-normalized C11,C12,C22, so scaling is unaffected.)
C_trace = C11 / C11(1) + C22 / C22(1);
fprintf('  C11(0)=%.5f  C22(0)=%.5f  (normalized trace used for detection)\n', ...
        C11(1), C22(1));
fprintf('  Lag window [0, %.1f] s,  Rayleigh ~ %.3f rad/s\n\n', tau(end), 2*pi/tau(end));

%% ---- HAMZSI PHASE 2: FREQUENCIES FROM TRACE ----------------------------
fprintf('Phase 2: Sparse spectral identification (trace, cosine basis)...\n');
[omega_identified, ~, coeffs, resnorm] = ...
    identify_modes(C_trace, Phi_cos, Omega_grid, tol_frac, dOmega_merge, dt);
nM = numel(omega_identified);
fprintf('  NNLS residual norm = %.6e\n', resnorm);
fprintf('  Active modes: %d\n', nM);
for j = 1:nM
    fprintf('    omega_%d_hat = %.4f rad/s\n', j, omega_identified(j));
end

%% ---- HAMZSI PHASE 3: MATRIX RESIDUES + COUPLING RECOVERY ---------------
fprintf('\nPhase 3: matrix residue recovery...\n');
w  = omega_identified(:);
B  = cos(tau * w');                                  % n_lag x nM
a11 = lsqnonneg(B, C11);                             % >=0 diagonal weights
a22 = lsqnonneg(B, C22);
a12 = B \ C12;                                       % off-diagonal may be signed
w2  = w.^2;
R_hat = cell(nM,1);  Rscaled = cell(nM,1);
for i = 1:nM
    Ri = [a11(i), a12(i); a12(i), a22(i)] * w2(i);   % de-tilted 2x2
    Ri = psd_project(Ri);                            % enforce R_i >= 0 (Thm 1)
    R_hat{i} = Ri;
end
G_true = cell(nM,1);  jmatch = zeros(nM,1);
for i = 1:nM
    [~, jm] = min(abs(omega_modes - w(i)));
    jmatch(i) = jm;  vj = Vobs(:, jm);
    G_true{i} = vj * vj';
end
num = 0; den = 0;
for i = 1:nM
    num = num + R_hat{i}(:)' * G_true{i}(:);
    den = den + R_hat{i}(:)' * R_hat{i}(:);
end
gscale = num/den;
errF = zeros(nM,1);
fprintf('  %8s | %25s | %25s | %8s\n','omega','R_hat (scaled)','R_true = v v^T','errF(%)');
for i = 1:nM
    Rs = gscale * R_hat{i};  Rscaled{i} = Rs;
    errF(i) = norm(Rs - G_true{i},'fro') / max(norm(G_true{i},'fro'), eps);
    fprintf('  %8.4f | [%.3f %.3f;%.3f %.3f] | [%.3f %.3f;%.3f %.3f] | %7.2f\n', ...
        w(i), Rs(1,1),Rs(1,2),Rs(2,1),Rs(2,2), ...
        G_true{i}(1,1),G_true{i}(1,2),G_true{i}(2,1),G_true{i}(2,2), 100*errF(i));
end
fprintf('  Mean residue-matrix error: %.2f%%   Max: %.2f%%\n', 100*mean(errF), 100*max(errF));

%% ---- FREQUENCY RECOVERY ACCURACY ---------------------------------------
fprintf('\nFrequency recovery comparison:\n');
fprintf('  %15s   %15s   %10s\n', 'True omega_i', 'Identified', 'Error (%)');
for j = 1:numel(omega_hidden_true)
    ow = omega_hidden_true(j);
    [~, im] = min(abs(omega_identified - ow));
    fprintf('  %15.4f   %15.4f   %10.2f\n', ow, omega_identified(im), ...
            100*abs(omega_identified(im)-ow)/ow);
end

%% ---- OUT-OF-SAMPLE FORECAST (both channels) ----------------------------
fit_frac = 0.40;  n_fit = round(fit_frac*Nt);
t_fit = t_vec(1:n_fit)';   wid = omega_identified(:)';
Reg_fit = [cos(t_fit*wid), sin(t_fit*wid)];
Reg_all = [cos(t_vec(:)*wid), sin(t_vec(:)*wid)];
ab1     = Reg_fit \ X_noisy(1:n_fit, 1);   x1_pred = Reg_all * ab1;
ab2     = Reg_fit \ X_noisy(1:n_fit, 2);   x2_pred = Reg_all * ab2;
x_pred  = x1_pred;                          % kept for any downstream use
t_split = t_vec(n_fit);

%% ---- PLOTTING -----------------------------------------------------------
figure('Name','B1: Spring-Mass Chain (2 outputs)','Color','w','Position',[50 50 950 680]);

win = (t_vec >= t_split-30) & (t_vec <= t_split+90);

% (a) Channel q1: data + out-of-sample forecast
subplot(2,2,1)
plot(t_vec(win), X_clean(win,1), 'b-', 'LineWidth', 1.2); hold on;
plot(t_vec(win), x1_pred(win), '--', 'Color',[0.85 0.4 0.0], 'LineWidth', 1.4);
yl = ylim;
patch([t_split-30 t_split t_split t_split-30],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.85 0.9 1.0],'EdgeColor','none','FaceAlpha',0.35);
xline(t_split, ':k','LineWidth',1.2);
text(t_split-26, yl(2)*0.82,'fit','Interpreter','latex','FontSize',7);
text(t_split+6,  yl(2)*0.82,'forecast','Interpreter','latex','FontSize',7);
xlabel('Time $t$ (s)','Interpreter','latex'); ylabel('$q_1(t)$ (m)','Interpreter','latex');
title('(a) Channel $q_1$: forecast vs truth','Interpreter','latex');
legend({'True $q_1$','Forecast $\hat{q}_1$'},'Interpreter','latex','FontSize',8);
xlim([t_split-30, t_split+90]); grid on; box on; set(gca,'FontSize',9);

% (b) Channel q4: data + out-of-sample forecast
subplot(2,2,2)
plot(t_vec(win), X_clean(win,2), '-','Color',[0.1 0.6 0.3], 'LineWidth', 1.2); hold on;
plot(t_vec(win), x2_pred(win), '--', 'Color',[0.85 0.4 0.0], 'LineWidth', 1.4);
yl = ylim;
patch([t_split-30 t_split t_split t_split-30],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.88 0.96 0.9],'EdgeColor','none','FaceAlpha',0.4);
xline(t_split, ':k','LineWidth',1.2);
text(t_split-26, yl(2)*0.82,'fit','Interpreter','latex','FontSize',7);
text(t_split+6,  yl(2)*0.82,'forecast','Interpreter','latex','FontSize',7);
xlabel('Time $t$ (s)','Interpreter','latex'); ylabel('$q_4(t)$ (m)','Interpreter','latex');
title('(b) Channel $q_4$: forecast vs truth','Interpreter','latex');
legend({'True $q_4$','Forecast $\hat{q}_4$'},'Interpreter','latex','FontSize',8);
xlim([t_split-30, t_split+90]); grid on; box on; set(gca,'FontSize',9);

% (c) Residue recovery: recovered vs true entries (11 and 12)
subplot(2,2,3)
for j = 1:numel(omega_hidden_true)
    xline(omega_hidden_true(j),'--','Color',[1 0.6 0.6],'LineWidth',0.8); hold on;
end
r11 = cellfun(@(R) R(1,1), Rscaled);  g11 = cellfun(@(G) G(1,1), G_true);
r12 = cellfun(@(R) R(1,2), Rscaled);  g12 = cellfun(@(G) G(1,2), G_true);
stem(w, g11, 'k^', 'MarkerSize',7,'LineWidth',1.0); hold on;
stem(w, r11, 'b', 'filled','MarkerSize',5);
stem(w, g12, 'ks', 'MarkerSize',6,'LineWidth',1.0);
stem(w, r12, 'Color',[0.1 0.6 0.3],'Marker','o','MarkerFaceColor',[0.1 0.6 0.3],'MarkerSize',4);
xlabel('Frequency $\omega$ (rad/s)','Interpreter','latex');
ylabel('Residue entries','Interpreter','latex');
title('(c) Matrix residue recovery','Interpreter','latex');
legend({'True $\omega_i^*$','$R^{true}_{11}$','$R^{rec}_{11}$','$R^{true}_{12}$','$R^{rec}_{12}$'}, ...
       'Interpreter','latex','FontSize',7,'Location','northeast');
xlim([0, 2.6]); grid on; box on; set(gca,'FontSize',9);

% (d) Coupling-matrix recovery: recovered vs true entries, grouped by mode.
% Each mode contributes three independent entries (R11, R12, R22; R21=R12);
% plotted as true/recovered pairs so agreement is read directly off heights.
subplot(2,2,4)
nMshow = numel(Rscaled);
Tvals = [];  Rvals = [];
for i = 1:nMshow
    T = G_true{i};  R = Rscaled{i};
    Tvals = [Tvals, T(1,1), T(1,2), T(2,2)]; %#ok<AGROW>
    Rvals = [Rvals, R(1,1), R(1,2), R(2,2)]; %#ok<AGROW>
end
ngrp = numel(Tvals);
xb = 1:ngrp;
hb = bar(xb, [Tvals(:), Rvals(:)], 1.0); hold on;
hb(1).FaceColor = [0.45 0.45 0.45];  hb(1).EdgeColor = 'none';   % true
hb(2).FaceColor = [0.20 0.40 0.85];  hb(2).EdgeColor = 'none';   % recovered
% light separators between modes (every 3 entries)
for i = 1:nMshow-1
    xline(3*i+0.5, ':', 'Color',[0.7 0.7 0.7]);
end
% Place the per-mode frequency labels as real x-ticks at the group centers;
% MATLAB positions them below the axis automatically.
grp_centers = 3*(0:nMshow-1) + 2;
set(gca, 'XTick', grp_centers, ...
         'XTickLabel', arrayfun(@(i) sprintf('$\\omega_%d$',i), 1:nMshow, ...
                                'UniformOutput', false), ...
         'TickLabelInterpreter','latex', 'FontSize', 9);
ylabel('Residue entry','Interpreter','latex');
title('(d) Coupling matrices: recovered vs.\ true','Interpreter','latex');
legend({'True','Recovered'},'Interpreter','latex','FontSize',8,'Location','northeast');
grid on; box on;
text(0.02, 0.93, sprintf('mean Frobenius error %.1f\\%%',100*mean(errF)), ...
     'Interpreter','latex','Units','normalized','FontSize',8);

sgtitle('Benchmark 1: Spring--Mass Chain ($N=4$, observe $q_1,q_4$, fixed--free)', ...
        'Interpreter','latex','FontSize',11,'FontWeight','bold');
% Use exportgraphics for clean vector text (R2020a+); fall back to print.
if exist('exportgraphics','file')
    exportgraphics(gcf,'Fig1_SpringMassChain.pdf','ContentType','vector');
else
    set(gcf,'Renderer','painters');
    print(gcf,'Fig1_SpringMassChain.pdf','-dpdf','-painters','-bestfit');
end
fprintf('\nFigure saved: Fig1_SpringMassChain.pdf\n');

%==========================================================================
%  LOCAL FUNCTIONS
%==========================================================================
function [X, q_all] = sim_trajectory(V, omega_modes, obs_idx, t_vec, Lq, Lv)
% One free (UNFORCED) trajectory from a single CONSTANT physical initial
% condition. ICs are drawn in PHYSICAL coordinates from the equilibrium
% distribution  q0 ~ N(0, K^{-1}),  v0 ~ N(0, M^{-1})  via Cholesky factors
% Lq, Lv. The chain is then released and evolves freely:
%     q(t) = V * eta(t),   eta_j(t) = eta0_j cos(w_j t) + (deta0_j/w_j) sin(w_j t),
% with modal ICs eta0 = V' q0, deta0 = V' v0 (V is M-orthonormal, M=I).
% Equipartition is NOT imposed here; it emerges from averaging C over many
% such runs (the equilibrium covariance above gives <E_j> equal across modes).
    N   = numel(omega_modes);
    q0  = Lq * randn(N,1);             % physical initial displacement (constant)
    v0  = Lv * randn(N,1);             % physical initial velocity     (constant)
    eta0  = V' * q0;                   % modal projection
    deta0 = V' * v0;
    om  = max(omega_modes, 1e-10);
    COSm = cos(om * t_vec);            % N x Nt
    SINm = sin(om * t_vec);
    ETA  = (eta0 .* COSm) + ((deta0 ./ om) .* SINm);   % N x Nt
    q_all = (V * ETA)';                % Nt x N  (free evolution)
    X     = q_all(:, obs_idx);         % Nt x n_obs
end

function c = xcorr_biased(a, b, n_lag)
% Biased cross-correlation <a(t+tau) b(t)>, lags 0..n_lag-1, mean+trend removed.
    a = detrend(a(:));  b = detrend(b(:));  Nt = numel(a);
    c = zeros(n_lag,1);
    for k = 0:n_lag-1
        c(k+1) = mean(a(k+1:Nt) .* b(1:Nt-k));
    end
end

function M = psd_project(M)
% Project a symmetric 2x2 onto the PSD cone (clip negative eigenvalues).
    M = (M + M')/2;
    [Q,L] = eig(M);  L = diag(max(diag(L),0));
    M = Q*L*Q';
end

function [omega_id, R_id, coeffs, resnorm] = ...
         identify_modes(C, Phi_cos, Omega_grid, tol_frac, dOmega_merge, dt)
% Non-negative cosine fit of a scalar C(tau), peak clustering, joint refine.
    Cn = C / C(1);
    [coeffs, resnorm] = lsqnonneg(Phi_cos, Cn);
    omega_id = [];  R_id = [];
    if all(coeffs==0), return; end
    active = find(coeffs > tol_frac*max(coeffs));
    if isempty(active), return; end
    grp_start = 1;
    for ii = 2:numel(active)
        if Omega_grid(active(ii)) - Omega_grid(active(ii-1)) > dOmega_merge
            [omega_id,R_id] = append_cluster(omega_id,R_id,active(grp_start:ii-1),Omega_grid,coeffs);
            grp_start = ii;
        end
    end
    [omega_id,R_id] = append_cluster(omega_id,R_id,active(grp_start:end),Omega_grid,coeffs);
    [omega_id, R_id] = refine_joint(Cn, omega_id, dt, 6);
end

function [omega, A] = refine_joint(C, omega0, dt, n_iter)
% Block coordinate descent: amplitudes (linear) + per-mode leave-one-out refine.
    C = C(:);  tau = (0:numel(C)-1)' * dt;
    omega = omega0(:);  nM = numel(omega);
    M = cos(tau * omega');
    for it = 1:n_iter
        A = M \ C;
        for j = 1:nM
            other = [1:j-1, j+1:nM];
            if isempty(other), rj = C; else, rj = C - M(:,other)*A(other); end
            ws = linspace(max(omega(j)-0.05,1e-3), omega(j)+0.05, 401);
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

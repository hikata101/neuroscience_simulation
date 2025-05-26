% Izhikevich_MAIN.m
% -------------------------------------------------------------
% Created by Eugene M. Izhikevich (orig. Python ver.)
% MATLAB translation 2025-05-27
% -------------------------------------------------------------
clear; clc; close all;

% 1- NETWORK SIZE:
Ne=800; Ni=200; % Excitatory, inhibitory. Ne+Ni is total neurons.
SIM_TIME=3000;

% 2 - GLOBAL PARAMETERS THAT SET OUR NEURON MODEL. DEFAULT IS SPIKING
% NEURON:
% Set initial conditions of neurons, with some variability provided by the
% vectors re and ri containing random numbers between 0 and 1.
re=rand(Ne,1); ri=rand(Ni,1);
a=[0.02*ones(Ne,1); 0.02+0.08*ri]; % Model parameters. Do not touch.
b=[0.2*ones(Ne,1); 0.25-0.05*ri];
c=[-65+15*re.^2; -65*ones(Ni,1)];
d=[8-6*re.^2; 2*ones(Ni,1)];

% 3 - SET UP THE CONNECTIVITY MATRIX: DIRECTED NETWORK
% In this construction, 1=connection exist, 0=no connection.
% Connectivity is set as random. Then, a fraction of connections are set 0.
% Note that effectively this is an Erd�s-R�nyi graphs, with no spatial
% characteristics. % If you want to symmetrize the network to make 
% undirected networks (not realistic in neuroscience), use A = (A + A.')/2
frac_delete=0.8; % Set this fraction of connections to zero.
A=[rand(Ne+Ni)];
A(A<frac_delete)=0;
A(A>0)=1;
A = A - diag(diag(A)); % Make the diagonal elements 0 (no self-connection).
firstA = A;                   % save copy

%% 4 ── SYNAPTIC WEIGHTS
MAX_EXC_WEIGHT = 4;
MAX_INH_WEIGHT = 0.5;
W_exc = MAX_EXC_WEIGHT * rand(Ne+Ni, Ne);
W_inh = -MAX_INH_WEIGHT * rand(Ne+Ni, Ni);
W     = [W_exc , W_inh];

%% 5 ── FINAL WEIGHTED CONNECTIVITY  S = A .* W
S = A .* W;

%% 6 ── NOISE LEVEL
NOISE_MAX = 3;

%% ──────────────────────────────────────────────────────────────
%  MAIN SIMULATION ROUTINES
%% ──────────────────────────────────────────────────────────────
main_simulation(A, SIM_TIME, 'original');

%% ===== motif matrices (3-,4-,5-node) =========================
motif_3a = [0 1 1; 1 0 1; 1 1 0];
motif_3b = [0 1 1; 1 0 1; 0 0 0];
motif_3c = [0 1 1; 0 0 0; 0 0 0];
motif_3d = [0 0 1; 1 0 1; 0 0 0];

motif_4a = [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0];
motif_4b = [0 1 1 0; 1 0 1 0; 1 1 0 1; 0 0 0 0];
motif_4c = [0 1 1 1; 1 0 1 1; 0 0 0 0; 0 0 0 0];
motif_4d = [0 0 1 1; 0 0 1 1; 0 0 0 0; 0 0 0 0];
motif_4e = [0 1 1 1; 0 0 0 0; 0 0 0 0; 0 0 0 0];
motif_4f = [0 0 1 1; 0 0 0 0; 1 0 0 0; 1 0 0 0];

motif_5a = [0 1 1 1 1; 1 0 1 1 1; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0];
motif_5b = [0 1 1 1 0; 1 0 1 1 1; 0 0 0 0 0; 0 0 0 0 0; 1 1 1 0 0];
motif_5c = [0 1 1 1 1; 1 0 1 1 1; 0 0 0 0 0; 0 0 0 0 0; 1 1 1 1 0];
motif_5d = [0 0 1 1 0; 0 0 1 1 1; 1 1 0 1 1; 1 1 1 0 1; 0 1 1 1 0];
motif_5e = [0 0 0 0 0; 1 0 0 0 0; 0 0 0 1 1; 1 0 1 0 1; 0 0 1 1 0];

all_motifs = {motif_3a, motif_3b, motif_3c, motif_3d, ...
              motif_4a, motif_4b, motif_4c, motif_4d, motif_4e, motif_4f, ...
              motif_5a, motif_5b, motif_5c, motif_5d, motif_5e};

all_motifs_names = { ...
    'motif_3a','motif_3b','motif_3c','motif_3d', ...
    'motif_4a','motif_4b','motif_4c','motif_4d','motif_4e','motif_4f', ...
    'motif_5a','motif_5b','motif_5c','motif_5d','motif_5e'};

%% ── run every motif once (optional)
% simulation_all_motifs(all_motifs, all_motifs_names, A);

%% ── example: combine 3 motifs in given percentages
combination_motifs       = {motif_3a, motif_4a, motif_5a};
combination_motifs_names = {'motif_3a','motif_4a','motif_5a'};
percent_motifs           = [0.3, 0.3, 0.4];  % must sum to 1
name_combination = sprintf('%0.1fm%s_%0.1fm%s_%0.1fm%s_', ...
    percent_motifs(1), combination_motifs_names{1}, ...
    percent_motifs(2), combination_motifs_names{2}, ...
    percent_motifs(3), combination_motifs_names{3});

comb_A   = A;                 % start from base connectivity
Nt       = Ne + Ni;
neurons  = randperm(Nt);      % shuffled list
steps    = floor(percent_motifs .* Nt);
idx      = 1;

for k = 1:numel(combination_motifs)
    motif      = combination_motifs{k};
    motif_size = size(motif,1);
    max_steps  = floor(steps(k) / motif_size);

    for s = 1:max_steps
        sel             = neurons(idx:idx+motif_size-1);
        comb_A(sel,sel) = motif;
        idx             = idx + motif_size;
    end
end

% main_simulation(comb_A, SIM_TIME, name_combination);
simulation_all_motifs(all_motifs, all_motifs_names, comb_A);

%% =============================================================
% ---------------------  SUB-FUNCTIONS  -------------------------
% =============================================================
function main_simulation(A, SIM_TIME, name_motif)
    % global parameters (shared with parent via nested func OR pass as arg)
    persistent Ne Ni a b c d S NOISE_MAX
    if isempty(Ne)         % copy from base workspace on first call
        Ne         = evalin('base','Ne');
        Ni         = evalin('base','Ni');
        a          = evalin('base','a');
        b          = evalin('base','b');
        c          = evalin('base','c');
        d          = evalin('base','d');
        S          = evalin('base','S');
        NOISE_MAX  = evalin('base','NOISE_MAX');
    end

    % --- visualise connectivity -------------------------------
    figure('Name',['Connectivity ' name_motif],'Color','w');
    imagesc(A); colormap(gray); axis equal tight;
    xlabel('Neuron'); ylabel('Neuron');
    title(['Connectivity matrix : ' strrep(name_motif,'_','\_')]);
    drawnow;

    % --- initialise state variables ---------------------------
    v        = -65 * ones(Ne+Ni,1);
    u        = b .* v;
    firings  = [];                      % [t i] pairs

    % --- simulation loop --------------------------------------
    for t = 1:SIM_TIME
        I = [ NOISE_MAX * randn(Ne,1) ; 2 * randn(Ni,1) ];

        fired = find(v >= 30);
        if ~isempty(fired)
            firings = [firings ; t * ones(numel(fired),1) , fired];
            v(fired) = c(fired);
            u(fired) = u(fired) + d(fired);
            I        = I + sum(S(:,fired),2);    % post-syn currents
        end

        % 0.5-ms step ×2  (≡ 1-ms Euler)
        v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);
        v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);
        u = u + a .* (b .* v - u);
    end

    % --- raster plot ------------------------------------------
    figure('Name',['Raster ' name_motif],'Color','w');
    scatter(firings(:,1), firings(:,2), 6, 'k', '.');
    xlabel('Time (ms)'); ylabel('Neuron index');
    title(['Raster plot : ' strrep(name_motif,'_','\_')]);
    drawnow;
end

% -------------------------------------------------------------
function simulation_all_motifs(all_motifs, all_names, A)
    SIM_TIME = evalin('base','SIM_TIME');
    for m = 1:numel(all_motifs)
        motif = all_motifs{m};
        name  = all_names{m};
        Nt    = size(A,1);
        newA  = A;
        neurons = randperm(Nt);
        motif_size = size(motif,1);
        max_steps  = floor(Nt / motif_size);
        for s = 1:max_steps
            sel            = neurons((s-1)*motif_size + (1:motif_size));
            newA(sel,sel)  = motif;
        end
        main_simulation(newA, SIM_TIME, name);
    end
end

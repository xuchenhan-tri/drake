clear; close all; clc;

%% Old force data (from Simon)
% Time horizon: 0–0.2 s
tvec = 0:0.0025:0.2;

% Old forces (components)
f_x = [0	-7.6349	-10.764	-12.0158	-11.9678	-10.989	-9.8167	-9.0442	-8.9121	-9.3102	-9.9189	-10.4033	-10.5753	-10.4393	-10.139	-9.8548	-9.7122	-9.7387	-9.8776	-10.0357	-10.1356	-10.1472	-10.0886	-10.005	-9.9413	-9.9213	-9.9426	-9.9844	-10.0224	-10.0401	-10.0348	-10.0151	-9.9937	-9.9809	-9.9802	-9.9888	-10.0002	-10.0086	-10.0108	-10.0076	-10.0017	-9.9967	-9.9945	-9.9955	-9.9982	-10.0011	-10.0027	-10.0027	-10.0015	-9.9999	-9.9988	-9.9986	-9.9991	-9.9999	-10.0005	-10.0008	-10.0006	-10.0002	-9.9999	-9.9997	-9.9997	-9.9999	-10.0001	-10.0002	-10.0002	-10.0002	-10.0001	-10	-9.9999	-10	-10	-10.0001	-10.0001	-10.0001	-10.0001	-10	-10	-10	-10	-10	-10];
f_y = 1e-13 .* [0	-0.00263678	0.198452	0.0548173	-0.0380251	-0.0437844	0.0170003	0.0911424	-0.0120737	0.123304	0.230718	0.257294	0.347361	0.095271	0.0675154	-0.0738298	0.0773687	0.0711931	0.0446865	0.065295	0.03407	0.0251882	0.121361	-0.0169309	-0.0371925	-0.0126982	0.0167921	0.01249	0.0294903	-0.00478784	0.0243555	-0.0247025	0.0268535	-0.195746	-0.172362	-0.0967976	-0.139194	-0.209693	-0.0997119	0.0131839	-0.0278944	-0.053707	-0.105957	-0.0544356	0.0837871	0.0351108	0.0214412	-0.0283454	-0.0330291	0.0471151	0.103216	-0.0200534	-0.0956873	-0.0580785	-0.096971	-0.164035	-0.171321	-0.0396211	0.0453457	-0.0431946	-0.0490233	0.00392048	0.0883668	0.0510009	0.0189779	-0.111681	-0.0231759	0.113624	0.0491968	0.0827463	0.138604	0.185615	0.139298	0.0506539	0.0465947	-0.0451028	-0.195885	-0.133053	-0.11758	-0.0694236	-0.172501];
f_z = [0	-1.37091	-2.04558	-2.45942	-2.61827	-2.52724	-2.27854	-1.98592	-1.7584	-1.65554	-1.67779	-1.78816	-1.92741	-2.0421	-2.10058	-2.09765	-2.04966	-1.98382	-1.92664	-1.89513	-1.89338	-1.9144	-1.94533	-1.97325	-1.9896	-1.99178	-1.98265	-1.96819	-1.95466	-1.9464	-1.94484	-1.94875	-1.95541	-1.96183	-1.96591	-1.96687	-1.96523	-1.96222	-1.95922	-1.95724	-1.95666	-1.95731	-1.95863	-1.96003	-1.961	-1.96134	-1.9611	-1.96051	-1.95985	-1.95937	-1.95917	-1.95925	-1.9595	-1.9598	-1.96003	-1.96014	-1.96012	-1.96001	-1.95988	-1.95976	-1.95969	-1.95969	-1.95974	-1.9598	-1.95986	-1.95989	-1.9599	-1.95984	-1.95979	-1.95977	-1.95977	-1.95979	-1.95982	-1.95984	-1.95985	-1.95985	-1.95984	-1.95982	-1.95981	-1.9598	-1.95979];

%% Plot old forces with solid curves in blue, orange, and green
figure;
hold on

oldLineWidth = 5;  % Thicker lines for visibility

% Colors for old forces:
% f_x: Blue ("#0072BD"), f_y: Orange ("#D95319"), f_z: Green ("#77AC30")
h1 = plot(tvec, f_x, 'LineWidth', oldLineWidth, 'Color', "#0072BD");
h2 = plot(tvec, f_y, 'LineWidth', oldLineWidth, 'Color', "#D95319");
h3 = plot(tvec, f_z, 'LineWidth', oldLineWidth, 'Color', "#77AC30");

%% Import new force data from "force_data.txt"
% The file is assumed to have 10000 rows and 3 columns (f_x, f_y, f_z)
newData = load('force_data.txt');  % Ensure this file is in the current folder

% Only take the first 2000 rows, and downsample a bit to match resolution
% of old data
newData = newData(1:5:2000, :);

% Create a time vector spanning 0–0.2 s for the new data
tvec_new = linspace(0, 0.2, size(newData, 1));

% Extract new force components
new_f_x = newData(:, 1);
new_f_y = newData(:, 2);
new_f_z = newData(:, 3);

%% Plot new forces with solid curves in red, purple, and black
newLineWidth = 5;  % Thicker lines for new forces

% Colors for new forces:
% f_x: Red ("#FF0000"), f_y: Purple ("#7E2F8E"), f_z: Black ("#000000")
h4 = plot(tvec_new, new_f_x, 'LineWidth', newLineWidth, 'Color', "#FF0000");
h5 = plot(tvec_new, new_f_y, 'LineWidth', newLineWidth, 'Color', "#7E2F8E");
h6 = plot(tvec_new, new_f_z, 'LineWidth', newLineWidth, 'Color', "#000000");

%% Create a two-column legend
% We want each row to represent one force component:
% Row 1: f_x (old) and f_x (new)
% Row 2: f_y (old) and f_y (new)
% Row 3: f_z (old) and f_z (new)
%
% To achieve this, we rearrange the legend handles accordingly.
legendHandles = [h1, h2, h3, h4, h5, h6];
legendLabels  = {'$f_x$ (old)', '$f_y$ (old)', ...
                 '$f_z$ (old)', '$f_x$ (new)', ...
                 '$f_y$ (new)', '$f_z$ (new)'};
leg = legend(legendHandles, legendLabels, ...
    'Location', 'best', 'Interpreter', 'latex', 'NumColumns', 2);
set(leg, 'FontSize', 32);

%% Format the figure
% Label the axes with larger font size
xlabel('Time [s]', 'FontSize', 32, 'Interpreter', 'latex');
ylabel('Force [N]', 'FontSize', 32, 'Interpreter', 'latex');

% Set x-axis limits to display the full time range
xlim([0 0.2]);
ylim([-20 3]);
% Increase tick label font size
set(gca, 'FontSize', 28, 'TickLabelInterpreter', 'latex');

% Resize the figure window to ensure all elements are visible
fig = gcf;
fig.Units = 'pixels';
fig.Position = [100 100 1400 1000];  % [left bottom width height]

%% Optional: add text annotations (adjust positions if needed)
x_limits = xlim;
x_position = x_limits(2) + 0.01 * (x_limits(2) - x_limits(1));  % Slightly right of x-axis limit

text(x_position, -1.962, '$mg/2$', ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
    'FontSize', 38, 'Interpreter', 'latex', 'Color', 'k');

text(x_position, -10, '$-f_e$', ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
    'FontSize', 38, 'Interpreter', 'latex', 'Color', 'k');

hold off


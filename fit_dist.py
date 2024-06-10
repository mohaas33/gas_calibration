
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))

hv_sets = []
mean_sets = []
sigma_sets = []
lin_gain = []


for gas in range(7):

    means = []
    sigmas = []
    hv_set = []

    if gas==0:
        gas_name = "Ar_CF4_60_40"
        hv_start = 4350
        NFiles = 8
        # List of CSV files (assuming they are named file1.csv, file2.csv, ..., file10.csv)
        files = [f'./TPC-Canary-2024/{hv_start+50*i}-1000.csv' for i in range(NFiles)]
        hv_set = [hv_start+50*i for i in range(NFiles)]
        AMax = 0.21

    if gas==1:
        gas_name = "Ar_CF4_Iso_75_20_5"
        hv_start = 3150
        NFiles = 7
        # List of CSV files (assuming they are named file1.csv, file2.csv, ..., file10.csv)
        files = [f'./TPC-Canary-2024/ArCF4Iso_75_20_5/{hv_start+50*i}_iso_000.csv' for i in range(NFiles)]
        hv_set = [hv_start+50*i for i in range(NFiles)]
        AMax = 0.31

    if gas==2:
        gas_name = "Ar_CF4_Iso_80_15_5"
        hv_start = 3025
        NFiles = 8
        #dv = [0,50,50,25,25,25,25]
        files = [f'./TPC-Canary-2024/ArCF4Iso_80_15_5_060624/{hv_start+25*i}_iso15_000.csv' for i in range(NFiles)]
        hv_set = [hv_start+25*i for i in range(NFiles)]
        AMax = 0.26

    if gas==3:
        gas_name = "Ar_CF4_Iso_85_10_5"
        hv_start = 2875
        NFiles = 7
        files = [f'./TPC-Canary-2024/ArCF4Iso_85_10_5/{hv_start+25*i}_iso10_000000.csv' for i in range(NFiles)]
        hv_set = [hv_start+25*i for i in range(NFiles)]
        AMax = 0.31
    if gas==4:
        gas_name = "ArCF4N2_80_10_10"
        hv_start = 4100
        NFiles = 7
        files = [f'./TPC-Canary-2024/ArCF4N2_80_10_10/{hv_start+50*i}_N10_000.csv' for i in range(NFiles)]
        hv_set = [hv_start+50*i for i in range(NFiles)]
        AMax = 0.31
    if gas==5:
        gas_name = "ArCF4N2_65_25_10"
        hv_start = 4400
        NFiles = 7
        files = [f'./TPC-Canary-2024/ArCF4N2_65_25_10/{hv_start+50*i}_N10_000.csv' for i in range(NFiles)]
        files.append("./TPC-Canary-2024/ArCF4N2_65_25_10/5000_N10_000.csv")
        hv_set = [hv_start+50*i for i in range(NFiles)]
        hv_set.append(5000)
        AMax = 0.31
    if gas==6:
        gas_name = "ArCF4Iso_75_20_5_060724"
        hv_start = 3250
        NFiles = 5
        files = [f'./TPC-Canary-2024/ArCF4Iso_75_20_5_060724/{hv_start+100*i}_iso_000.csv' for i in range(NFiles)]
        files.append("./TPC-Canary-2024/ArCF4Iso_75_20_5_060724/3750_LinGain4_iso_000.csv")
        files.append("./TPC-Canary-2024/ArCF4Iso_75_20_5_060724/3850_LinGain1_iso_000.csv")
        files.append("./TPC-Canary-2024/ArCF4Iso_75_20_5_060724/3950_LinGain1_iso_000.csv")
        files.append("./TPC-Canary-2024/ArCF4Iso_75_20_5_060724/4000_LinGain1_iso_000.csv")
        hv_set = [hv_start+100*i for i in range(NFiles)]
        hv_set.append(3750)
        hv_set.append(3850)
        hv_set.append(3950)
        hv_set.append(4000)
        lin_gain = [1] * NFiles
        lin_gain.append(10.0/4)
        lin_gain.append(10.0/1)
        lin_gain.append(10.0/1)
        lin_gain.append(10.0/1)
        AMax = 2.31


    fig, axes = plt.subplots(2, 4, figsize=(12, 9))
    if gas == 6:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    fig.suptitle('Fits for different voltages')

    for k,file in enumerate(files):
        print(file)
        # Read the CSV file
        data = pd.read_csv(file, header=None)
        data.columns = ['x', 'y']

        # Extract the 'x' and 'y' values
        x = data['x']
        y = data['y']
        #if gas==6:
        #    x = x * lin_gain[k]
        # Create a weighted histogram of the 'x' values
        hist, bin_edges = np.histogram(x, bins=50, weights=y, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ## Fit a Gaussian distribution to the weighted histogram
        #mu, sigma = norm.fit(np.repeat(bin_centers, hist.astype(int)))
        # Initial guess for the parameters
        initial_guess = [ -0.08069917, 12.57593375,  0.03150833]
        if gas==6 and k>6:
            initial_guess = [ -2, 12.57593375,  1.1150833]

        # Fit the Gaussian to the data
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=popt)
        print(popt)
        # Extract the fitted parameters
        mu, amplitude, sigma = popt      
        
        i=int(k/4)
        j=k-i*4
        print(i, " ", j)
        axes[i,j].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='black')
        # Plot the PDF.
        xmin = mu - 5*sigma
        xmax = mu + 5*sigma
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)

        axes[i,j].plot(x, p, 'k', linewidth=2)
        axes[i,j].set_xlim(-AMax,0)
        if gas==5 and k==7:
            axes[i,j].set_xlim(-7*AMax,0)

        if i>0:
            axes[i,j].set_xlabel('Amplitude [V]')

        textstr = '\n'.join((
        r'$\mu=%.4f$' % (mu, ),
        r'$\sigma=%.4f$' % (sigma, ),
        r'$HV=%d$' % (hv_set[k], )))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        axes[i,j].text(0.05, 0.95, textstr, transform=axes[i,j].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        # Store the mean
        means.append(mu)

        sigmas.append(sigma)

        #hv_set.append(hv_start-300+50*k)

    fig.savefig('./Plots/Fits_{}.pdf'.format(gas_name))
    
    # Convert lists to numpy arrays for easier indexing
    means = np.array(means)
    sigmas = np.array(sigmas)
    hv_set = np.array(hv_set)

    if gas==6:
        means = means * lin_gain[k]
        sigmas = sigmas * lin_gain[k]

    hv_sets.append(hv_set)
    mean_sets.append(means)
    sigma_sets.append(100*(sigmas/means))
    

    # Create a 2D scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(hv_set, -means, c=sigmas, cmap='viridis', marker='o')
    line = ax.plot(hv_set, -means, linestyle='-', color='b')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sigma')

    # Add labels and title
    plt.ylabel('Mean of Gaussian Fit')
    plt.xlabel('HV Setting - 300 [V]')
    plt.title('Means of Gaussian Fits for Each Setting')

    plt.grid(True)
    plt.savefig('./Plots/HV_vs_Mean_{}.pdf'.format(gas_name))
    #plt.show()

# Create a 2D scatter plot
fig_set, ax_set = plt.subplots(figsize=(8, 6))
scatter0 = ax_set.scatter(hv_sets[0], -mean_sets[0], marker='o', label='Ar:CF4 (60/40)')
scatter1 = ax_set.scatter(hv_sets[1], -mean_sets[1], marker='o', label='Ar:CF4:Iso (75/20/5)')
scatter2 = ax_set.scatter(hv_sets[2], -mean_sets[2], marker='o', label='Ar:CF4:Iso (80/15/5)')
scatter3 = ax_set.scatter(hv_sets[3], -mean_sets[3], marker='o', label='Ar:CF4:Iso (85/10/5)')
scatter4 = ax_set.scatter(hv_sets[4], -mean_sets[4], marker='o', label='Ar:CF4:N2 (80/10/10)')
scatter5 = ax_set.scatter(hv_sets[5], -mean_sets[5], marker='X', label='Ar:CF4:N2 (65/25/10)')
scatter6 = ax_set.scatter(hv_sets[6], -mean_sets[6], marker='X', label='Ar:CF4:Iso (75/20/5)')
#line = ax_set.plot(hv_set, -means, linestyle='-', color='b')
plt.grid(True)
# Add labels and title
plt.ylabel('Mean of Gaussian Fit')
plt.xlabel('HV Setting - 300 [V]')
plt.title('Means of Gaussian Fits for All Setting')
plt.xlim(2600,5250)
# Function add a legend
plt.legend(loc = (0.05, 0.7))#"center")

plt.savefig('./Plots/HV_vs_Mean_set.pdf')
plt.ylim(0,0.20)
plt.savefig('./Plots/HV_vs_Mean_set_zoom.pdf')

#plt.show()

fig_set_s, ax_set_s = plt.subplots(figsize=(8, 6))
scatter0_s = ax_set_s.scatter(hv_sets[0], -sigma_sets[0], marker='o', label='Ar:CF4 (60/40)')
scatter1_s = ax_set_s.scatter(hv_sets[1], -sigma_sets[1], marker='o', label='Ar:CF4:Iso (75/20/5)')
scatter2_s = ax_set_s.scatter(hv_sets[2], -sigma_sets[2], marker='o', label='Ar:CF4:Iso (80/15/5)')
scatter3_s = ax_set_s.scatter(hv_sets[3], -sigma_sets[3], marker='o', label='Ar:CF4:Iso (85/10/5)')
scatter4_s = ax_set_s.scatter(hv_sets[4], -sigma_sets[4], marker='o', label='Ar:CF4:N2 (80/10/10)')
scatter5_s = ax_set_s.scatter(hv_sets[5], -sigma_sets[5], marker='X', label='Ar:CF4:N2 (65/25/10)')
scatter6_s = ax_set_s.scatter(hv_sets[6], -sigma_sets[6], marker='X', label='Ar:CF4:Iso (75/20/5)')
#line = ax_set.plot(hv_set, -means, linestyle='-', color='b')
plt.grid(True)
# Add labels and title
plt.ylabel(r'$\sigma / \mu [\%]$')
plt.xlabel('HV Setting - 300 [V]')

plt.title('Sigma Gaussian Fits for All Setting')

# Function add a legend
plt.legend()

plt.savefig('./Plots/HV_vs_Sigma_set.pdf')


plt.show()



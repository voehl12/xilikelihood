import numpy as np
import matplotlib.pyplot as plt
from calc_pdf import get_combs
from scipy.interpolate import UnivariateSpline  # Import UnivariateSpline
from file_handling import read_posterior_files

numjobs = 30
num_croco = 10
num_auto = 5
num_angular_bins = 3
filestring_croco = "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_1000sqd_fiducial_nonoise_1dcomb_*.npz"
filestring_auto = "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_1000sqd_fiducial_nonoise_1dcomb_*.npz"

# Use regex to filter files for auto and croco
posteriors_croco = read_posterior_files(filestring_croco, regex=r'1dcomb_\d+_croco\.npz')
posteriors_auto = read_posterior_files(filestring_auto, regex=r'1dcomb_\d+_auto\.npz')

s8_croco = posteriors_croco["s8"]
s8_auto = posteriors_auto["s8"]
gauss_posteriors_croco = posteriors_croco["gauss_posteriors"]
gauss_posteriors_auto = posteriors_auto["gauss_posteriors"]
exact_posteriors_croco = posteriors_croco["exact_posteriors"]
exact_posteriors_auto = posteriors_auto["exact_posteriors"]
means_croco = posteriors_croco["means"]
means_auto = posteriors_auto["means"]
combs_croco = posteriors_croco["combs"]

combs_auto = posteriors_auto["combs"]
available = posteriors_croco["available"]

# Count occurrences of redshift bin combinations using numpy
unique_crocos, counts_crocos = np.unique(combs_croco[:,0], axis=0, return_counts=True)
unique_autos, counts_autos = np.unique(combs_auto[:,0], axis=0, return_counts=True)
# Combine counts from both croco and auto
counts = np.concatenate((counts_crocos, counts_autos))
if len(unique_crocos) != num_croco:
    print("Number of unique croco combinations does not match expected.")
if len(unique_autos) != num_auto:
    print("Number of unique auto combinations does not match expected.")

# Check if all counts equal num_angular_bins
if np.all(counts == num_angular_bins):
    print("All redshift bin combinations appear the correct number of times.")
else:
    print("Some redshift bin combinations do not appear the correct number of times.")

#s8 = np.linspace(0.6, 0.9, 100)

ang_bins = [(0.4541123210836613, 1.010257752338312), (1.010257752338312, 2.247507232845216), (2.247507232845216, 5.000000000000002)]
# Normalize the posteriors
angs = [np.mean(angbin) for angbin in ang_bins]
croco_nums = [2,4,5,7,8,9,11,12,13,14]
num_angular_bins = 3
num_redshift_bins = 5
# Set up the plot
fig, axes = plt.subplots(num_angular_bins, num_croco, figsize=(15, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

# Plot each subplot
n = 0
a = 0
for i in range(num_angular_bins):
    for j in range(num_croco):
        if available[a]:
            
            print(n,a)
            curr_s8 = s8_croco[n]
            normalized_gauss_post = gauss_posteriors_croco[n]
            normalized_post = exact_posteriors_croco[n]
            mean_gauss = means_croco[n,1]
            mean_exact = means_croco[n,0]
            rs_bin = combs_croco[n,0]
            ang_bin = combs_croco[n,1]
            croco = get_combs(rs_bin)
            croco_ind = np.argwhere(croco_nums == rs_bin)[0][0]
            print(croco_ind)
            ax = axes[ang_bin, croco_ind]
            ax.plot(curr_s8, normalized_gauss_post, color="red", label="Gaussian")
            ax.axvline(mean_gauss,color='red')
            ax.plot(curr_s8, normalized_post, color="blue", label="Exact")
            ax.axvline(mean_exact,color='blue')
            #ax.set_xlim(0.5,1.1)
            #croco = combs_croco[n,0]
            
            ang_bin_tuple = ang_bins[ang_bin]
            textstr = '$\overline{{\\theta}} = {:.2f}^{{\circ}} - {:.2f}^{{\circ}}$\n$n_z = ({:d},{:d})$'.format(ang_bin_tuple[0], ang_bin_tuple[1],croco[0]+1,croco[1]+1)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=props)
            
            # Only show x-axis labels on the bottom row
            if ang_bin == num_angular_bins - 1:
                ax.set_xlabel("s8")
            

            
            # Only show y-axis labels on the first column
            if rs_bin == 0:
                ax.set_ylabel('Posterior')
            else:
                ax.set_yticklabels([])

            if ang_bin == 0 and croco_ind == 0:
                ax.legend(loc='upper right')
            
            n += 1
        
        a += 1

fig.savefig('all_1d_posts_croco_1000sqd_new.png')

# Determine the largest s8 range across both auto and cross-correlations
s8_min = min(min(s[0] for s in s8_croco), min(s[0] for s in s8_auto))
s8_max = max(max(s[-1] for s in s8_croco), max(s[-1] for s in s8_auto))
common_s8 = np.linspace(s8_min, s8_max, 50000)  # Define a common grid with 50000 points

# Interpolate all posteriors to the common s8 grid using lower-order splines
interpolated_gauss_posteriors_croco = []
interpolated_exact_posteriors_croco = []
interpolated_gauss_posteriors_auto = []
interpolated_exact_posteriors_auto = []

for i in range(len(gauss_posteriors_croco)):
    spline_gauss = UnivariateSpline(s8_croco[i], gauss_posteriors_croco[i], k=3, s=0)
    spline_exact = UnivariateSpline(s8_croco[i], exact_posteriors_croco[i], k=3, s=0)
    interpolated_gauss_posteriors_croco.append(spline_gauss(common_s8))
    interpolated_exact_posteriors_croco.append(spline_exact(common_s8))

for i in range(len(gauss_posteriors_auto)):
    spline_gauss = UnivariateSpline(s8_auto[i], gauss_posteriors_auto[i], k=3, s=0)
    spline_exact = UnivariateSpline(s8_auto[i], exact_posteriors_auto[i], k=3, s=0)
    interpolated_gauss_posteriors_auto.append(spline_gauss(common_s8))
    interpolated_exact_posteriors_auto.append(spline_exact(common_s8))

interpolated_gauss_posteriors_croco = np.array(interpolated_gauss_posteriors_croco)
interpolated_exact_posteriors_croco = np.array(interpolated_exact_posteriors_croco)
interpolated_gauss_posteriors_auto = np.array(interpolated_gauss_posteriors_auto)
interpolated_exact_posteriors_auto = np.array(interpolated_exact_posteriors_auto)

# Filter posteriors for angular bin 3 (combs[i,1] == 2) for both auto and cross-correlations
filtered_gauss_posteriors_croco = []
filtered_exact_posteriors_croco = []
filtered_gauss_posteriors_auto = []
filtered_exact_posteriors_auto = []

for i in range(len(combs_croco)):
    print(combs_croco[i])
    if combs_croco[i, 1] == 2:
        filtered_gauss_posteriors_croco.append(interpolated_gauss_posteriors_croco[i])
        filtered_exact_posteriors_croco.append(interpolated_exact_posteriors_croco[i])
print("Filtered auto posteriors:")
for i in range(len(combs_auto)):
    print(combs_auto[i])
    if combs_auto[i, 1] == 2:
        filtered_gauss_posteriors_auto.append(interpolated_gauss_posteriors_auto[i])
        filtered_exact_posteriors_auto.append(interpolated_exact_posteriors_auto[i])

filtered_gauss_posteriors_croco = np.array(filtered_gauss_posteriors_croco)
filtered_exact_posteriors_croco = np.array(filtered_exact_posteriors_croco)
filtered_gauss_posteriors_auto = np.array(filtered_gauss_posteriors_auto)
filtered_exact_posteriors_auto = np.array(filtered_exact_posteriors_auto)

# Compute the joint posterior for the filtered posteriors (both auto and cross-correlations)
joint_gauss_posterior_croco = np.prod(filtered_gauss_posteriors_croco, axis=0)
joint_exact_posterior_croco = np.prod(filtered_exact_posteriors_croco, axis=0)
joint_gauss_posterior_auto = np.prod(filtered_gauss_posteriors_auto, axis=0)
joint_exact_posterior_auto = np.prod(filtered_exact_posteriors_auto, axis=0)

# Normalize the joint posteriors
joint_gauss_posterior_croco /= np.trapz(joint_gauss_posterior_croco, common_s8)
joint_exact_posterior_croco /= np.trapz(joint_exact_posterior_croco, common_s8)
joint_gauss_posterior_auto /= np.trapz(joint_gauss_posterior_auto, common_s8)
joint_exact_posterior_auto /= np.trapz(joint_exact_posterior_auto, common_s8)

# Compute the means of the joint posterior
mean_joint_gauss_croco = np.trapz(common_s8 * joint_gauss_posterior_croco, common_s8)
mean_joint_exact_croco = np.trapz(common_s8 * joint_exact_posterior_croco, common_s8)
mean_joint_gauss_auto = np.trapz(common_s8 * joint_gauss_posterior_auto, common_s8)
mean_joint_exact_auto = np.trapz(common_s8 * joint_exact_posterior_auto, common_s8)

# Combine filtered auto and croco posteriors (angular bin 3 only)
filtered_all_gauss_posteriors = np.concatenate((filtered_gauss_posteriors_croco, filtered_gauss_posteriors_auto), axis=0)
filtered_all_exact_posteriors = np.concatenate((filtered_exact_posteriors_croco, filtered_exact_posteriors_auto), axis=0)

# Compute the joint posterior for the filtered posteriors (all, auto, and croco)
joint_gauss_posterior_all = np.prod(filtered_all_gauss_posteriors, axis=0)
joint_exact_posterior_all = np.prod(filtered_all_exact_posteriors, axis=0)

# Normalize the joint posteriors
joint_gauss_posterior_all /= np.trapz(joint_gauss_posterior_all, common_s8)
joint_exact_posterior_all /= np.trapz(joint_exact_posterior_all, common_s8)

# Compute the means of the joint posterior for all
mean_joint_gauss_all = np.trapz(common_s8 * joint_gauss_posterior_all, common_s8)
mean_joint_exact_all = np.trapz(common_s8 * joint_exact_posterior_all, common_s8)

# Plot the joint posterior with means for all, auto, and croco (filtered angular bin 3)
fig_joint_filtered, ax_joint_filtered = plt.subplots(figsize=(8, 6))
colors = ['blue', 'green', 'orange']
# Plot for all
color = colors[0]
ax_joint_filtered.plot(common_s8, joint_gauss_posterior_all, color=color, linestyle=':',label="Gaussian (All)")
ax_joint_filtered.axvline(mean_joint_gauss_all, color=color, linestyle=":")
ax_joint_filtered.plot(common_s8, joint_exact_posterior_all, color=color, label="Exact (All)")
ax_joint_filtered.axvline(mean_joint_exact_all, color=color, linestyle="-")

# Plot for auto
color = colors[1]
ax_joint_filtered.plot(common_s8, joint_gauss_posterior_auto, color=color, linestyle=':', label="Gaussian (Auto)")
ax_joint_filtered.axvline(mean_joint_gauss_auto, color=color, linestyle=':')
ax_joint_filtered.plot(common_s8, joint_exact_posterior_auto, color=color, label="Exact (Auto)")
ax_joint_filtered.axvline(mean_joint_exact_auto, color=color, linestyle="-")

# Plot for croco
color = colors[2]
ax_joint_filtered.plot(common_s8, joint_gauss_posterior_croco, color=color, linestyle=':', label="Gaussian (Cross)")
ax_joint_filtered.axvline(mean_joint_gauss_croco, color=color, linestyle=':')
ax_joint_filtered.plot(common_s8, joint_exact_posterior_croco, color=color, label="Exact (Cross)")
ax_joint_filtered.axvline(mean_joint_exact_croco, color=color, linestyle="-")


# Finalize plot
ax_joint_filtered.set_xlabel("s8")
ax_joint_filtered.set_ylabel("Posterior")
ax_joint_filtered.set_xlim(0.6, 1.0)
ax_joint_filtered.legend(loc="upper right")
ax_joint_filtered.set_title("Joint posterior from multiplied 1d likelihoods, large scales")
fig_joint_filtered.savefig('s8posterior_frommarginals_largescales_1000sqd_new.png')

# Plot the joint posterior with means for both auto and cross-correlations
fig_joint, ax_joint = plt.subplots(figsize=(8, 6))
ax_joint.plot(common_s8, joint_gauss_posterior_croco, color="red", label="Joint Gaussian (Cross)")
ax_joint.axvline(mean_joint_gauss_croco, color="red", linestyle="--", label="Mean Gaussian (Cross)")
ax_joint.plot(common_s8, joint_exact_posterior_croco, color="blue", label="Joint Exact (Cross)")
ax_joint.axvline(mean_joint_exact_croco, color="blue", linestyle="--", label="Mean Exact (Cross)")
ax_joint.plot(common_s8, joint_gauss_posterior_auto, color="orange", label="Joint Gaussian (Auto)")
ax_joint.axvline(mean_joint_gauss_auto, color="orange", linestyle="--", label="Mean Gaussian (Auto)")
ax_joint.plot(common_s8, joint_exact_posterior_auto, color="green", label="Joint Exact (Auto)")
ax_joint.axvline(mean_joint_exact_auto, color="green", linestyle="--", label="Mean Exact (Auto)")
ax_joint.set_xlabel("s8")
ax_joint.set_ylabel("Posterior")
ax_joint.set_xlim(0.6, 1.0)
ax_joint.legend(loc="upper right")
ax_joint.set_title("Joint Posterior Comparison (Auto and Cross)")
fig_joint.savefig('joint_1d_post_auto_cross_1000sqd_new.png')

# Plot all interpolated posteriors
fig_all, ax_all = plt.subplots(figsize=(10, 6))
for i in range(len(interpolated_gauss_posteriors_croco)):
    ax_all.plot(common_s8, interpolated_gauss_posteriors_croco[i], color="red", alpha=0.5, label="Gaussian (Cross)" if i == 0 else "")
    ax_all.plot(common_s8, interpolated_exact_posteriors_croco[i], color="blue", alpha=0.5, label="Exact (Cross)" if i == 0 else "")

for i in range(len(interpolated_gauss_posteriors_auto)):
    ax_all.plot(common_s8, interpolated_gauss_posteriors_auto[i], color="orange", alpha=0.5, label="Gaussian (Auto)" if i == 0 else "")
    ax_all.plot(common_s8, interpolated_exact_posteriors_auto[i], color="green", alpha=0.5, label="Exact (Auto)" if i == 0 else "")

ax_all.set_xlabel("s8")
ax_all.set_ylabel("Posterior")
ax_all.set_title("All Interpolated Posteriors")
ax_all.legend(loc="upper right")
ax_all.set_xlim(0.75,0.85)
ax_all.set_ylim(0, 40)
fig_all.savefig('all_interpolated_posts_croco_1000sqd_new.png')
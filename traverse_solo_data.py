import scipy.io
import os
import numpy as np
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import pickle
import scipy.stats as stats


def load_SoloData_mats(datadir):
    all_mat_files = []  # List of mat files (dicts)
    all_file_names = []  # list of file names of mat files ordered by when read
    # Traverse through sub-directories and read mat files in to all_mat_files
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if file.endswith('.mat'):
                print(files)
                mat_struct = scipy.io.loadmat(os.path.join(root, file))
                all_mat_files.append(mat_struct['saved'])
                all_file_names.append(files)

    rat_name_entries = []
    for entry in all_mat_files:
        rat_name_entries.append(entry['SavingSection_ratname'][0][0][0])
    rat_names = set(rat_name_entries)

    rat_name_entries = np.array(rat_name_entries)
    rat_dict = {}
    for rat in rat_names:
        ind_list = []
        name_idx = rat_name_entries == rat
        for i in zip(name_idx, all_mat_files):
            if i[0] == True:
                ind_list.append(i[1])
        rat_dict[rat] = ind_list
    return rat_dict


R_trials_fields = ['StimulusSection_nTrialsClass1',
                   'StimulusSection_nTrialsClass2',
                   'StimulusSection_nTrialsClass3',
                   'StimulusSection_nTrialsClass4']

L_trials_fields = ['StimulusSection_nTrialsClass5',
                   'StimulusSection_nTrialsClass6',
                   'StimulusSection_nTrialsClass7',
                   'StimulusSection_nTrialsClass8']

field_names = ['SideSection_CP_duration']

# rat_dict = load_SoloData_mats(r'H:\Rat_Data')
rat_dict = pickle.load(open(r'H:\Rat_Data\Analysis_Data\rat_dict.pkl', 'rb'))
# Write rat dictionary to pickle
f = open(r'H:\Rat_Data\Analysis_Data\rat_dict.pkl','wb')
pickle.dump(rat_dict,f)
f.close()
rat_mean_dict = {}
rat_ntrials = {}

# Extract total CP durations for each rat
for rat in rat_dict:
    cp_dur_sess = []
    for session in rat_dict[rat]:
        cp_dur_field = field_names[0]
        cp_dur = session[cp_dur_field][0][0][0][0]
        if cp_dur > 0.05:
            cp_dur_sess.append(cp_dur)
    rat_mean_dict[rat] = np.array(cp_dur_sess)

# Extract Total number of trials for each rat
for rat in rat_dict:
    ntrial_sess = []
    for session in rat_dict[rat]:
        trial_fields = L_trials_fields+R_trials_fields
        all_side_classes = 0
        for field in trial_fields:
            ntrial = session[field][0][0][0][0]
            all_side_classes += ntrial
        ntrial_sess.append(all_side_classes)
    rat_ntrials[rat] = np.array(ntrial_sess)

sample_0_names = [
                  'VP01','VP02',
                  'AA03','AA04',
                  'DO03','DO04',
                  'DO05','DO06',
                  'SC05','SC06',
                  'VP05','VP06'
                  ]
sample_1_names = [
                  'AA01','AA02',
                  'DO01','DO02',
                  'SC01','SC02',
                  'SC02','SC04',
                  'VP03','VP04',
                  'AA05','AA06',
                  'AA07','AA08',
                  'DO07','DO08',
                  'VP07','VP08'
                   ]
sample_0_cps = []
sample_1_cps = []
sample_0_N = []
sample_1_N = []

for rat in sample_0_names:
    sample_0_cps.extend(rat_mean_dict[rat])
    sample_0_N.extend(rat_ntrials[rat])

for rat in sample_1_names:
    sample_1_cps.extend(rat_mean_dict[rat])
    sample_1_N.extend(rat_ntrials[rat])
# do t test
print('CP comp', ttest_ind(np.array(sample_0_cps), np.array(sample_1_cps),equal_var=False))
print('N trials comp', ttest_ind(np.array(sample_0_N), np.array(sample_1_N),equal_var=False))

# ANOVA
print(stats.f_oneway(sample_0_cps,sample_1_cps))
print(stats.f_oneway(sample_0_N,sample_1_N))


fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax1.hist((np.array(sample_0_cps), np.array(sample_1_cps)),
         bins=5,color=('r', 'b'), label=['Delay in Watering','No Delay'],density=True)

ax2 = fig1.add_subplot(212)
ax2.hist((np.array(sample_0_N),np.array(sample_1_N)),
         bins=5,color=('r','b'), label=['Delay in Watering','No Delay'],density=True)
ax1.set_xlabel('Centre Poke duration')
ax2.set_xlabel('Number of Trials')
ax1.set_ylabel('Number')
ax2.set_ylabel('Number')
ax1.legend()
ax2.legend()

"""
regression model evaluation 
 - load training data for BP positions at 0 and 5 mm BP 
 - combine data from both datasets for prediction 
 - predict based on the models 
 - plot the energy and position  errors 

energy and positions errors are based on alexander code
https://github.com/SiFi-CC/SiFiCC-SplitNeuralNetwork/blob/main/src/SiFiCCNN/plotting/plt_evaluation.py 
run the functions at the end of this scripts before plotting 
or move them to before 

saving the MC True data is done in the script: pos_energy_array_training
"""
#%%
#from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.utils import class_weight
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
#%%
path =  "path"

input_data_BP0mm = path+r"training_data_name_BP0mm"
input_data_BP5mm = path+r"\training_data_name_BP5mm"

input_data_BP0mm = np.load(input_data_BP0mm)
input_data_BP5mm = np.load(input_data_BP5mm)

e_pos_BP0mm_name = "e_pos_BP0mm_name" 
e_pos_BP5mm_name = "e_pos_BP5mm_name" 
p_pos_BP0mm_name = "p_pos_BP0mm_name" 
p_pos_BP5mm_name = "p_pos_BP5mm_name" 
## ep means electron and photon energies 
ep_energy_BP0mm_name = "e_energy_BP0mm_name" 
ep_energy_BP5mm_name = "p_energy_BP5mm_name" 


e_pos_BP0mm = np.load(path+e_pos_BP0mm_name)
p_pos_BP0mm = np.load(path+p_pos_BP0mm_name)
energy_ep_BP0mm = np.load(path+ep_energy_BP0mm_name)

e_pos_BP5mm = np.load(path+e_pos_BP5mm_name)
p_pos_BP5mm = np.load(path+p_pos_BP5mm_name)
energy_ep_BP5mm = np.load(path+ep_energy_BP5mm_name)
#%%
input_data_BP0mm = input_data_BP0mm['arr_0']#.swapaxes(1,2)
input_data_BP5mm = input_data_BP5mm['arr_0']#.swapaxes(1,2)


e_pos_BP0mm = e_pos_BP0mm["arr_0"] 
e_pos_BP5mm = e_pos_BP5mm["arr_0"] 

p_pos_BP0mm = p_pos_BP0mm["arr_0"]
p_pos_BP5mm = p_pos_BP5mm["arr_0"]

energy_ep_BP0mm = energy_ep_BP0mm["arr_0"]
energy_ep_BP5mm = energy_ep_BP5mm["arr_0"]

pos_BP0mm = np.concatenate([e_pos_BP0mm,p_pos_BP0mm],axis=1)
pos_BP5mm = np.concatenate([e_pos_BP5mm,p_pos_BP5mm],axis=1)

pos_ep = np.concatenate([pos_BP0mm,pos_BP5mm])
energy_ep = np.concatenate([energy_ep_BP0mm, energy_ep_BP0mm])
#%%
input_data_BP5mm.shape

#%%
## load model 

#model_name =r"C:\Users\georg\Desktop\master_thesis\Models\fourth_model\best_model_pos_regreession_5mmBP_fullnorm.h5" 
model_path = r"model path"
model_folder=  r"\increasing_0404_3216_norm26k_pos_regression_8128layers"
best_model = r"\best_model_pos_regression_shuffeldBP_step001_layernorm.h5"
model_pos = keras.models.load_model(model_path+model_folder+best_model)

model_folder=  r"model path"
best_model = r"\best_model_energy_regression_shuffeldBP_step001_layernorm.h5"
model_energy = keras.models.load_model(model_path+model_folder+best_model)

# %%
# slice data

## slice to validation and train 

trainset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.6)
trainset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.6)

valset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.8)
valset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.8)


X_train_BP0mm = input_data_BP0mm[:trainset_index_BP0mm]
X_train_BP5mm = input_data_BP5mm[:trainset_index_BP5mm]

X_val_BP0mm = input_data_BP0mm[valset_index_BP0mm:]
X_val_BP5mm = input_data_BP5mm[valset_index_BP5mm:]


X_test_BP0mm = input_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
X_test_BP5mm = input_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]

Y_train_pos_BP0mm = pos_BP0mm[:trainset_index_BP0mm]
Y_train_pos_BP5mm = pos_BP5mm[:trainset_index_BP5mm]


Y_val_pos_BP0mm = pos_BP0mm[valset_index_BP0mm:]
Y_val_pos_BP5mm = pos_BP5mm[valset_index_BP5mm:]

Y_test_pos_BP0mm = pos_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
Y_test_pos_BP5mm = pos_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]

Y_train_energy_BP0mm = energy_ep_BP0mm[:trainset_index_BP0mm]
Y_train_energy_BP5mm = energy_ep_BP5mm[:trainset_index_BP5mm]

Y_val_energy_BP0mm = energy_ep_BP0mm[valset_index_BP0mm:]
Y_val_energy_BP5mm = energy_ep_BP5mm[valset_index_BP5mm:]

Y_test_energy_BP0mm = energy_ep_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
Y_test_energy_BP5mm = energy_ep_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]
#%%
## Combine two data sets of BP0mm and BP5mm
X_test = np.concatenate([X_test_BP0mm,X_test_BP5mm])
Y_test_pos = np.concatenate([Y_test_pos_BP0mm,Y_test_pos_BP5mm])
Y_test_energy = np.concatenate([Y_test_energy_BP0mm,Y_test_energy_BP5mm])
#%%
Y_val_test_energy = np.concatenate([Y_test_energy_BP0mm,Y_val_energy_BP0mm,Y_test_energy_BP5mm,Y_val_energy_BP5mm])
Y_val_test_pos = np.concatenate([Y_test_pos_BP0mm,Y_val_pos_BP0mm,Y_test_pos_BP5mm,Y_val_pos_BP5mm])
X_val_test = np.concatenate([X_test_BP0mm,X_val_BP0mm,X_test_BP5mm,X_val_BP5mm])

X_all = np.concatenate([input_data_BP0mm,input_data_BP5mm])
Y_all_energy = np.concatenate([energy_ep_BP0mm,energy_ep_BP5mm])
Y_all_pos = np.concatenate([pos_BP0mm,pos_BP5mm])

#%%
## predict different data sets 
## the data sets all and valtest are used for reconstruction 
## the data set test is used for regression errors
Y_pred_pos = model_pos.predict(X_test)
Y_pred_energy = model_energy.predict(X_test)

#%%
Y_pred_pos_val_test = model_pos.predict(X_val_test)
Y_pred_energy_val_test = model_energy.predict(X_val_test)
#%%
Y_pred_pos_all = model_pos.predict(X_all)
Y_pred_energy_all = model_energy.predict(X_all)

#%%
## predict whole data set
Y_pred_energy_BP0mm = Y_pred_energy[0:len(Y_test_energy_BP0mm)]
Y_pred_energy_BP5mm = Y_pred_energy[len(Y_test_energy_BP0mm):]

Y_pred_pos_BP0mm = Y_pred_pos[0:len(Y_test_pos_BP0mm)]
Y_pred_pos_BP5mm = Y_pred_pos[len(Y_test_pos_BP0mm):]

#%%
## energy and position errors
### run the functions below first to run this 
plot_energy_error(Y_pred_energy_val_test,Y_val_test_energy,r"C:\Users\georg\Desktop\master_thesis\SiPM-NN-EventSelection\regression_analysis_valtest\energy_error_both11")
plot_position_error(Y_pred_pos_val_test,Y_val_test_pos,r"C:\Users\georg\Desktop\master_thesis\SiPM-NN-EventSelection\regression_analysis_valtest\pos_error_both")
plot_energy_error(Y_pred_energy_all,Y_all_energy,r"C:\Users\georg\Desktop\master_thesis\SiPM-NN-EventSelection\regression_analysis_all\energy_error_both")
plot_position_error(Y_pred_pos_all,Y_all_pos,r"C:\Users\georg\Desktop\master_thesis\SiPM-NN-EventSelection\regression_analysis_all\pos_error_both")

#%%
save_path = "save_path"
## saving output 
np.savez(save_path+r"\Y_pred_energy_val_test.npz",Y_pred_energy_val_test)
np.savez(save_path+r"\Y_pred_pos_val_test.npz",Y_pred_pos_val_test)
np.savez(save_path+r"\Y_true_energy_val_test.npz",Y_val_test_energy)
np.savez(save_path+r"\Y_true_pos_val_test.npz",Y_val_test_pos)
#%%
np.savez(save_path+r"\Y_pred_energy_test.npz",Y_pred_energy)
np.savez(save_path+r"\Y_pred_pos_test.npz",Y_pred_pos)
np.savez(save_path+r"\Y_true_energy_test.npz",Y_test_energy)
np.savez(save_path+r"\Y_true_pos_test.npz",Y_test_pos)
#%%
np.savez(save_path+r"\Y_pred_energy_all.npz",Y_pred_energy_all)
np.savez(save_path+r"\Y_pred_pos_all.npz",Y_pred_pos_all)
np.savez(save_path+r"\Y_energy_all.npz",Y_all_energy)
np.savez(save_path+r"\Y_pos_all.npz",Y_all_pos)
#%%

### ploting the loss 
## validation and train loss saved as csv in the training scripts 
loss_train_energy = np.loadtxt("energy_train_loss_csv")
loss_val_energy = np.loadtxt("energy_val_loss_csv")
loss_train_pos = np.loadtxt("pos_train_loss_csv")
loss_val_pos = np.loadtxt("pos_val_loss_csv")
#%%
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(7.5,5))
plt.plot(np.arange(0,len(loss_train_energy),1),loss_train_energy,c="k",label="train")
plt.plot(np.arange(0,len(loss_val_energy),1),loss_val_energy,c="k",ls="--",label="loss")
plt.grid()
plt.xlabel("epoch",fontsize=14)
plt.ylabel("loss",fontsize=14)
plt.title("Loss Energy MAE", fontsize=16)
plt.legend()
plt.savefig("loss_energy.png")
plt.show()
#%%
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(7.5,5))
plt.plot(np.arange(0,len(loss_train_pos),1),loss_train_pos,c="k",label="train")
plt.plot(np.arange(0,len(loss_val_pos),1),loss_val_pos,c="k",ls="--",label="loss")
plt.grid()
plt.xlabel("epoch",fontsize=14)
plt.ylabel("loss",fontsize=14)
plt.title("Loss Pos MAE", fontsize=16)
plt.legend()
plt.savefig("loss_pos.png")
plt.show()
#%%
######################################################
###### Code copied from Alexander #####################
########################################################

# fitting functions

def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


def max_super_function(x):
    return (0.1 + np.exp((0.5 * (x + 3)) / 2))/(1+np.exp((8*x+5)/3))/6

def plot_energy_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})
    width = 0.01
    bins_err = np.arange(-1.5, 1.5, width)
    bins_energy = np.arange(0.0, 10.0, width)

    bins_err_center = bins_err[:-1] + (width / 2)

    hist0, _ = np.histogram(y_pred[:, 0] - y_true[:, 0], bins=bins_err)
    hist1, _ = np.histogram(y_pred[:, 1] - y_true[:, 1], bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(lorentzian, bins_err_center, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(lorentzian, bins_err_center, hist1, p0=[0.0, 0.5, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron Energy Resolution")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$ [MeV]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, lorentzian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon Energy Resolution")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$ [MeV]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, lorentzian(ary_x, *popt1), color="green",
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt1[0], popt1[1] / 2))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$E_{Pred}$ - $E_{True}$ [MeV]")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0], bins=[bins_energy, bins_err], norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_relative.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$E_{Pred}$ - $E_{True}$ [MeV]")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1], bins=[bins_energy, bins_err], norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_relative.png")
    plt.close()


def plot_position_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    width = 0.1
    bins_err_x = np.arange(-5.5, 5.5, width)
    bins_err_y = np.arange(-60.5, 60.5, width)
    bins_err_z = np.arange(-5.5, 5.5, width)

    bins_x = np.arange(150.0 - 20.8 / 2.0, 270.0 + 46.8 / 2.0, width)
    bins_y = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
    bins_z = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)

    hist0, _ = np.histogram(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x)
    hist1, _ = np.histogram(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y)
    hist2, _ = np.histogram(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z)
    hist3, _ = np.histogram(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x)
    hist4, _ = np.histogram(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y)
    hist5, _ = np.histogram(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z)

    # fitting position resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist1, p0=[0.0, 20.0, np.sum(hist1) * width])
    popt2, pcov2 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist2, p0=[0.0, 1.0, np.sum(hist2) * width])
    popt3, pcov3 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist3, p0=[0.0, 1.0, np.sum(hist3) * width])
    popt4, pcov4 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist4, p0=[0.0, 20.0, np.sum(hist4) * width])
    popt5, pcov5 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist5, p0=[0.0, 1.0, np.sum(hist5) * width])

    ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
    ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
    ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Electron")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0], bins=[bins_x[:209], bins_err_x], norm=LogNorm())
    plt.xlim(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0)
    plt.hlines(xmin=150.0 - 20.8 / 2.0, xmax=150.0 + 20.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt1), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt1[0], popt1[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Electron")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt2), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt2[0], popt2[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Electron")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 2], y=y_pred[:, 2] - y_true[:, 2], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z_relative.png")
    plt.close()

    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt3), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt3[0], popt3[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Photon")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 3], y=y_pred[:, 3] - y_true[:, 3], bins=[bins_x[467:], bins_err_x], norm=LogNorm())
    plt.xlim(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0)
    plt.hlines(xmin=270.0 - 46.8 / 2.0, xmax=270.0 + 46.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt4), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt4[0], popt4[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Photon")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 4], y=y_pred[:, 4] - y_true[:, 4], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt5), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt5[0], popt5[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Photon")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 5], y=y_pred[:, 5] - y_true[:, 5], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z_relative.png")
    plt.close()
#%%


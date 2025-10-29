import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)


from CNN_multiclass_data_mergeSims_A100 import load_cnn_model #load_cnn_model
from GRL_multiclass_data_Simulations_A100 import load_grl_model_lambdaScheduler, intermediate_layer_model
from ImaGene_Phased_aDNA_SLiM_mh import * 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve, auc
from sklearn.preprocessing import label_binarize
#from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def compute_roc(y_true,y_pred,n_classes):
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   # Compute ROC curve
   for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

   return fpr,tpr,roc_auc



def compute_precision_recall(y_true,y_pred,n_classes):   
   precision = dict()
   recall = dict()
   pr_auc = dict()
   for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
      pr_auc[i] = auc(recall[i], precision[i])
   
   return precision,recall,pr_auc

def plotPRC(precision1, recall1, pr_auc1, model1_label,
            precision2, recall2, pr_auc2, model2_label,
            class_labels, colors, path):
    """
    Plots the Precision-Recall curves for two models on the same graph.
    Each class will have two lines (one for each model) of the same color
    but different linestyles.
    """
    plt.figure(figsize=(8, 6))
    linestyles = ['-', '--']  # Solid for model 1, dashed for model 2

    for i, class_label in enumerate(class_labels):
        # Plotting for the first model (e.g., single channel)
        plt.plot(recall1[i], precision1[i], color=colors[i], linestyle=linestyles[0], lw=2,
                 label=f'{model1_label} {class_label} (area = {pr_auc1[i]:0.2f})')

        # Plotting for the second model (e.g., multi-channel)
        plt.plot(recall2[i], precision2[i], color=colors[i], linestyle=linestyles[1], lw=2,
                 label=f'{model2_label} {class_label} (area = {pr_auc2[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig(path)
    plt.close()

def roc(class_labels_src,class_labels_tar,y_pred_matched,y_pred_missMatched,y_pred_grl,model_CNN,model_GRL):

   #get number of classes in data
   n_classes_src = 3+1 

   #combine hard + soft
   combine_src_sweeps = np.logical_or(class_labels_src[:, 1],class_labels_src[:, 2]).astype(int).reshape(-1, 1)
   combine_tar_sweeps = np.logical_or(class_labels_tar[:, 1],class_labels_tar[:, 2]).astype(int).reshape(-1, 1)

   y_true_src=np.hstack([class_labels_src, combine_src_sweeps])
   y_true_tar=np.hstack([class_labels_tar, combine_tar_sweeps])

   # probabilities 
   combine_matched_sweeps_pred = np.maximum(y_pred_matched[:, 1], y_pred_matched[:, 2]).reshape(-1, 1)
   combine_missMatched_sweeps_pred = np.maximum(y_pred_missMatched[:, 1], y_pred_missMatched[:, 2]).reshape(-1, 1)
   combine_grl_sweeps_pred = np.maximum(y_pred_grl[:, 1], y_pred_grl[:, 2]).reshape(-1, 1)

   y_pred_matched_combined = np.hstack([y_pred_matched, combine_matched_sweeps_pred])
   y_pred_missMatched_combined = np.hstack([y_pred_missMatched, combine_missMatched_sweeps_pred])
   y_pred_grl_combined = np.hstack([y_pred_grl, combine_grl_sweeps_pred])


    #Calcultate roc curves
   fpr_matched,tpr_matched,roc_auc_matched=compute_roc(y_true_src,y_pred_matched_combined,n_classes_src)
   fpr_missMatched,tpr_missMatched,roc_auc_missMatched=compute_roc(y_true_tar,y_pred_missMatched_combined,n_classes_src) # is the true for missmathched target data???
   fpr_grl,tpr_grl,roc_auc_grl=compute_roc(y_true_tar,y_pred_grl_combined,n_classes_src)
    
    #PLOT PRC
   class_labels = ["Neutral", "Hard sweep", "Soft sweep","Sweeps"]
   colors_CNN = ["#79706E", "#E15759", "#4E79A7","#B07AA1"]
   colors_GRL = ["#BAB0AC", "#FF9D9A", "#A0CBE8","#D4A6C8"]
   path =["roc_neutral.png","roc_HS.png","roc_SS.png","roc_sweep.png"]
   for i in range(n_classes_src):
      plt.figure(figsize=(8, 6))
      plt.plot(fpr_matched[i], tpr_matched[i], color=colors_CNN[i],linestyle='-', lw=2, label=f'AUC-PRC matched = {roc_auc_matched[i]:.2f}')
      plt.plot(fpr_missMatched[i], tpr_missMatched[i], color=colors_CNN[i],linestyle='--', lw=2, label=f'AUC-PRC misspecified= {roc_auc_missMatched[i]:.2f}')
      plt.plot(fpr_grl[i], tpr_grl[i], color=colors_GRL[i],linestyle='-', lw=2,  label=f'AUC-PRC GRL= {roc_auc_grl[i]:.2f}')
      plt.xlabel('FPR')
      plt.ylabel('TPR')
      plt.title(class_labels[i])
      plt.legend()
      plt.savefig(path[i])
      plt.close()
    

    


def precision_recall(class_labels_src,class_labels_tar,y_pred_matched,y_pred_missMatched,y_pred_grl,model_CNN,model_GRL):   

    #get number of classes in data
    n_classes_src = 3+1 

    #combine hard + soft
    combine_src_sweeps = np.logical_or(class_labels_src[:, 1],class_labels_src[:, 2]).astype(int).reshape(-1, 1)
    combine_tar_sweeps = np.logical_or(class_labels_tar[:, 1],class_labels_tar[:, 2]).astype(int).reshape(-1, 1)

    y_true_src=np.hstack([class_labels_src, combine_src_sweeps])
    y_true_tar=np.hstack([class_labels_tar, combine_tar_sweeps])
    

    # probabilities
    # 
    combine_matched_sweeps_pred = np.maximum(y_pred_matched[:, 1], y_pred_matched[:, 2]).reshape(-1, 1)
    combine_missMatched_sweeps_pred = np.maximum(y_pred_missMatched[:, 1], y_pred_missMatched[:, 2]).reshape(-1, 1)
    combine_grl_sweeps_pred = np.maximum(y_pred_grl[:, 1], y_pred_grl[:, 2]).reshape(-1, 1)

    y_pred_matched_combined = np.hstack([y_pred_matched, combine_matched_sweeps_pred])
    y_pred_missMatched_combined = np.hstack([y_pred_missMatched, combine_missMatched_sweeps_pred])
    y_pred_grl_combined = np.hstack([y_pred_grl, combine_grl_sweeps_pred])


    #Calcultate PR curves
    precision_matched,recall_matched,pr_auc_matched=compute_precision_recall(y_true_src,y_pred_matched_combined,n_classes_src)
    precision_missMatched,recall_missMatched,pr_auc_missMatched=compute_precision_recall(y_true_tar,y_pred_missMatched_combined,n_classes_src) # is the true for missmathched target data???
    precision_grl,recall_grl,pr_auc_grl=compute_precision_recall(y_true_tar,y_pred_grl_combined,n_classes_src)
    
    #PLOT PRC
    class_labels = ["Neutral", "Hard sweep", "Soft sweep","Sweeps"]
    colors_CNN = ["#79706E", "#E15759", "#4E79A7","#B07AA1"]
    colors_GRL = ["#BAB0AC", "#FF9D9A", "#A0CBE8","#D4A6C8"]
    path =["auprc_neutral.png","auprc_HS.png","auprc_SS.png","auprc_sweep.png"]
    for i in range(n_classes_src): #n_classes_src
      plt.figure(figsize=(8, 6))
      plt.rcParams.update({'font.size': 16})
      plt.plot(recall_matched[i], precision_matched[i], color=colors_CNN[i],linestyle='-', lw=2.5, label=f'AUC-PRC matched = {pr_auc_matched[i]:.2f}')
      plt.plot(recall_missMatched[i], precision_missMatched[i], color=colors_CNN[i],linestyle='--', lw=2.5, label=f'AUC-PRC misspecified= {pr_auc_missMatched[i]:.2f}')
      plt.plot(recall_grl[i], precision_grl[i], color=colors_GRL[i],linestyle='-', lw=2,  label=f'AUC-PRC GRL= {pr_auc_grl[i]:.2f}')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title(class_labels[i])
      plt.legend()
      plt.savefig(path[i])
      plt.close()

def confusionMatrix(class_labels,y_pred,output_names):
    #print(gene_sim_test.targets_classifier)
    output_predictions = {}
    for i in range(len(output_names)):
      output_predictions[output_names[i]] = y_pred[i]

    y_test_labels = np.argmax(class_labels, axis=1)

    print(np.unique(y_test_labels))
    print(np.unique(np.argmax(output_predictions['classifier'],axis=1)))

    cm = confusion_matrix(y_test_labels, np.argmax(output_predictions['classifier'],axis=1))
    accuracy = np.trace(cm) / float(np.sum(cm))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(3)#len(gene_sim_test.classes_classifier))
    plt.xticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    plt.yticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    thresh = cm.max() / 1.5
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.tight_layout()
    # Add labels to the confusion matrix plot
    for i in range(3):#len(gene_sim_test.classes_classifier)):
        for j in range(3):#len(gene_sim_test.classes_classifier)):
            plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.savefig('ConfusionMatrix.png')
    plt.close()

def confusionMatrix_CNN(class_labels,y_pred):
    #print(gene_sim_test.targets_classifier)

    y_test_labels = np.argmax(class_labels, axis=1)

    print(np.unique(y_test_labels))
    print(np.unique(np.argmax(y_pred,axis=1)))

    cm = confusion_matrix(y_test_labels, np.argmax(y_pred,axis=1))
    accuracy = np.trace(cm) / float(np.sum(cm))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(3)#len(gene_sim_test.classes_classifier))
    plt.xticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    plt.yticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    thresh = cm.max() / 1.5
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.tight_layout()
    # Add labels to the confusion matrix plot
    for i in range(3):#len(gene_sim_test.classes_classifier)):
        for j in range(3):#len(gene_sim_test.classes_classifier)):
            plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.savefig('ConfusionMatrix_CNN.png')
    plt.close()

def latent_visualization(model_GRL,model_CNN,layer_name,gene_sim_src,gene_sim_tar,class_labels_src,class_labels_tar):
   #Model that get desired layer
   inter_layer_modelGRL = intermediate_layer_model(model_GRL,layer_name)
   inter_layer_modelCNN = intermediate_layer_modelCNN(model_CNN,layer_name)

   num_samples = 1000


   # Randomly select indices
   #'''
   #sweep and neutral together
   random_src_indices = np.random.choice(gene_sim_src.shape[0], size=num_samples, replace=False)
   random_tar_indices = np.random.choice(gene_sim_tar.shape[0], size=num_samples, replace=False)

   #all neutral and sweep 
   layer_outputGRL_src = inter_layer_modelGRL.predict(gene_sim_src[random_src_indices])
   layer_outputGRL_tar = inter_layer_modelGRL.predict(gene_sim_tar[random_tar_indices])

   layer_outputCNN_src = inter_layer_modelCNN.predict(gene_sim_src[random_src_indices])
   layer_outputCNN_tar = inter_layer_modelCNN.predict(gene_sim_tar[random_tar_indices])

   all_featuresGRL = np.concatenate(( layer_outputGRL_src, layer_outputGRL_tar))


   #targets
   all_labelsGRL = np.concatenate((class_labels_src[random_src_indices],class_labels_tar[random_tar_indices]))#real label (sweep/neutral)
  
   all_featuresCNN = np.concatenate((layer_outputCNN_src, layer_outputCNN_tar))


#---------------------------------------------------------------------------------------------
## main

# 1. DEFINE PATHS AND PARAMETERS
path_src = "/u/project/ngarud/Garud_lab/DANN/DANNcolor/ProcessingData/ProcessedMayaSims/"
# --- MODIFIED: Provide paths for both models ---
model_bw_path = 'CNN_bw_multiclass_sims_trained'    # Path to your single-channel (B&W) model
model_color_path = 'CNN_color_multiclass_sims_trained' # <-- CHANGE THIS to the path for your color model
n_classes = 3

# 2. LOAD TEST DATA
print("Loading test data...")
# Load the full data, including all color channels
data_neu_src_full = np.load(path_src + "Neu_sims.npy", mmap_mode='r')[41000:46000,:,:,:]
data_hs_src_full = np.load(path_src + "HS_sims.npy", mmap_mode='r')[41000:46000,:,:,:]
data_ss_src_full = np.load(path_src + "SS_sims.npy", mmap_mode='r')[41000:46000,:,:,:]

# --- MODIFIED: Create datasets for both models ---
# Data for the color model (uses all channels)
X_test_color = np.concatenate((data_neu_src_full, data_hs_src_full, data_ss_src_full), axis=0)

# Data for the B&W model (uses only the first channel)
# We use [..., 0:1] to keep the channel dimension, resulting in a shape of (N, H, W, 1)
X_test_bw = X_test_color[..., 0:1] 

print(f"Test data shape (Color): {X_test_color.shape}")
print(f"Test data shape (B&W):   {X_test_bw.shape}")


# 3. CREATE TRUE LABELS (Y_test) - This is the same for both models
print("Creating true labels...")
neutral_label = np.array([1., 0., 0.])
HS_label = np.array([0., 1., 0.])
SS_label = np.array([0., 0., 1.])

y_true = np.concatenate((
    np.tile(neutral_label, (len(data_neu_src_full), 1)),
    np.tile(HS_label, (len(data_hs_src_full), 1)),
    np.tile(SS_label, (len(data_ss_src_full), 1))
))

# 4. LOAD MODELS AND MAKE PREDICTIONS
print("--- Processing Single-Channel (B&W) Model ---")
print(f"Loading model from: {model_bw_path}")
model_CNN_bw = load_cnn_model(model_bw_path)
print("Generating predictions for B&W model...")
y_pred_proba_bw = model_CNN_bw.predict(X_test_bw)

print("\n--- Processing Multi-Channel (Color) Model ---")
print(f"Loading model from: {model_color_path}")
model_CNN_color = load_cnn_model(model_color_path)
print("Generating predictions for color model...")
y_pred_proba_color = model_CNN_color.predict(X_test_color)


# 5. COMPUTE AND PLOT AUPRC
print("\nComputing and plotting combined AUPRC curves...")
# Compute metrics for the B&W model
precision_bw, recall_bw, pr_auc_bw = compute_precision_recall(y_true, y_pred_proba_bw, n_classes)

# Compute metrics for the color model
precision_color, recall_color, pr_auc_color = compute_precision_recall(y_true, y_pred_proba_color, n_classes)


# Define labels and colors for the plot
class_labels = ["Neutral", "Hard sweep", "Soft sweep"]
colors = ["#79706E", "#E15759", "#4E79A7"]

# --- MODIFIED: Use the new plotting function to compare models ---
plotPRC(precision_bw, recall_bw, pr_auc_bw, '1-Channel CNN',
        precision_color, recall_color, pr_auc_color, '2-Channel CNN',
        class_labels, colors, path='AUPRC_CNN_Comparison.png')

print("Done! Plot saved to AUPRC_CNN_Comparison.png")

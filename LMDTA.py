import random, os
from dataset import CustomDataSet, collate_fn
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from tensorboardX import SummaryWriter
import timeit
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index
import pickle
import pandas as pd

# Calculate Pearson correlation coefficient
def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

# Calculate Spearman correlation coefficient
def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

# Calculate R squared error
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


# Get scaling factor k for squared error zero calculation
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


# Calculate squared error zero
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


# Calculate RM2 metric
def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


# Model evaluation on test/validation/train set
def test_process(model, pbar):
    loss_f = nn.MSELoss()
    model.eval()
    test_losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for i, data in pbar:
            id, compounds, proteins, mol_embs, prot_embs, pair_tensor, labels = data
            id, compounds, proteins, mol_embs, prot_embs, pair_tensor, labels = id.cuda(), compounds.cuda(), proteins.cuda(), mol_embs.cuda(), prot_embs.cuda(), pair_tensor.cuda(), labels.cuda()
            predicts = model.forward(id, compounds, proteins,
                                     mol_embs, prot_embs, pair_tensor)

            loss = loss_f(predicts, labels.view(-1, 1))
            total_preds = torch.cat((total_preds, predicts.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
            test_losses.append(loss.item())
    
    Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
    test_loss = np.average(test_losses)
    return Y, P, test_loss, mean_squared_error(Y, P), mean_absolute_error(Y, P), r2_score(Y, P), concordance_index(Y,
                                                                                                                   P), rm2(
        Y, P), pearson(Y, P), spearman(Y, P)


# Wrapper for model evaluation and result saving
def test_model(test_dataset_load, save_path, DATASET, lable="Train", save=True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    T, P, loss_test, mse_test, mae_test, r2_test, ci_test, rm2_test, ps_test, sm_test = test_process(model, test_pbar)
    if save:
        with open(save_path + "/{}_stable_{}_prediction.txt".format(DATASET, lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};MSE:{:.5f};MAE:{:.5f};R2:{:.5f};CI:{:.5f};RM2:{:.5f};PS:{:.5f};SM:{:.5f}.' \
        .format(lable, loss_test, mse_test, mae_test, r2_test, ci_test, rm2_test, ps_test, sm_test)
    print(results)
    return results, mse_test, mae_test, r2_test, ci_test, rm2_test, ps_test, sm_test

import os

# Set CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    """select seed"""
    SEED = 3047
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """select dataset"""
    DATASETS=["Davis","Metz"]
    dataset_choice = 0
    DATASET = DATASETS[dataset_choice]
    pklDATASET = DATASETS[dataset_choice]

    # Model and embedding settings
    mol_models = ["ChemBERTa"]
    mol_lens = [94, 114, 512]
    mol_len = mol_lens[dataset_choice]
    mol_dim = 384

    pro_models = ["ESM-2-320"]
    pro_lens = [2551, 2529, 4130]
    pro_len = pro_lens[dataset_choice]
    pro_dim = 320

    print("Train in {}".format(DATASET))

    for mol_model in mol_models:
        for pro_model in pro_models:
            # Load train, valid, test datasets and embeddings
            tst_path = './datasets/{}/{}_train.csv'.format(DATASET, DATASET)
            trainset = pd.read_csv(tst_path)
            tst_path = './datasets/{}/pair_embedding/{}_train_encoder_max_id_tensor.pkl'.format(DATASET, pklDATASET)
            with open(tst_path, 'rb') as f:
                train_tensor = pickle.load(f)
            print("train load finished")

            tst_path = './datasets/{}/{}_valid.csv'.format(DATASET, DATASET)
            validset = pd.read_csv(tst_path)
            tst_path = './datasets/{}/pair_embedding/{}_valid_encoder_max_id_tensor.pkl'.format(DATASET, pklDATASET)
            with open(tst_path, 'rb') as f:
                valid_tensor = pickle.load(f)
            print("valid load finished")

            tst_path = './datasets/{}/{}_test.csv'.format(DATASET, DATASET)
            testset = pd.read_csv(tst_path)
            tst_path = './datasets/{}/pair_embedding/{}_test_encoder_max_id_tensor.pkl'.format(DATASET, pklDATASET)
            with open(tst_path, 'rb') as f:
                test_tensor = pickle.load(f)
            print("test load finished")

            tst_path = './datasets/{}/molecule_embedding/molecule_mapping_emb_{}.pkl'.format(DATASET, mol_model)
            with open(tst_path, 'rb') as f:
                mol_embedding = pickle.load(f)

            tst_path = './datasets/{}/protein_embedding/protein_mapping_emb_{}.pkl'.format(DATASET, pro_model)
            with open(tst_path, 'rb') as f:
                pro_embedding = pickle.load(f)

            # Training hyperparameters
            K_Fold = 5
            Batch_size = 4
            weight_decay = 1e-3
            lr = [5e-5]
            Patience = 300
            Epoch = 300
            for Learning_rate in lr:
                """Output files."""
                save_path = "./Results_{}_lmdta/".format(DATASET)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_results = save_path + 'The_results.txt'

                # Lists to store metrics for each fold
                MSE_List, MAE_List, R2_List, CI_List, RM2_List, PS_List, SM_List = [], [], [], [], [], [], []

                for i_fold in range(K_Fold):
                    # Prepare datasets and dataloaders
                    train_dataset = CustomDataSet(trainset, mol_embedding, pro_embedding, train_tensor)
                    valid_dataset = CustomDataSet(validset, mol_embedding, pro_embedding, valid_tensor)
                    test_dataset = CustomDataSet(testset, mol_embedding, pro_embedding, test_tensor)
                    train_size = len(train_dataset)
                    train_dataset_load = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=0,
                                                    collate_fn=collate_fn)
                    valid_dataset_load = DataLoader(valid_dataset, batch_size=Batch_size * 2, shuffle=False,
                                                    num_workers=0,
                                                    collate_fn=collate_fn)
                    test_dataset_load = DataLoader(test_dataset, batch_size=Batch_size * 2, shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=collate_fn)

                    """ create model"""
                    model = LMDTA(mol_len=mol_len, mol_dim=mol_dim, pro_len=pro_len, pro_dim=pro_dim).cuda()
                    """weight initialize"""
                    weight_p, bias_p = [], []
                    for p in model.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                    for name, p in model.named_parameters():
                        if 'bias' in name:
                            bias_p += [p]
                        else:
                            weight_p += [p]
                    LOSS_F = nn.MSELoss()
                    optimizer = optim.AdamW(
                        [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}],
                        lr=Learning_rate)
                    # Learning rate scheduler
                    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250, 300, 350, 400], gamma=0.6)
                    save_path_i = "{}/{}_Fold/".format(save_path, i_fold)
                    if not os.path.exists(save_path_i):
                        os.makedirs(save_path_i)
                    note = ""
                    writer = SummaryWriter(log_dir=save_path_i, comment=note)

                    """Start training."""
                    print('Training...')
                    start = timeit.default_timer()
                    patience = 0
                    test_best_epoch = 0
                    valid_best_epoch = 0
                    valid_best_score = 100
                    test_best_score = 100
                    for epoch in tqdm(range(1, Epoch + 1)):
                        # Training loop
                        trian_pbar = tqdm(
                            enumerate(
                                BackgroundGenerator(train_dataset_load)),
                            total=len(train_dataset_load))
                        train_losses_in_epoch = []
                        model.train()
                        for trian_i, train_data in trian_pbar:
                            '''data preparation '''
                            train_id, train_compound, train_protein, train_mol_embed, train_prot_embed, train_pair_embed, train_labels = train_data
                            train_id = train_id.cuda()
                            train_compound = train_compound.cuda()
                            train_protein = train_protein.cuda()
                            train_mol_embed = train_mol_embed.cuda()
                            train_prot_embed = train_prot_embed.cuda()
                            train_pair_embed = train_pair_embed.cuda()
                            train_labels = train_labels.cuda()
                            optimizer.zero_grad()


                            predicts = model.forward(train_id, train_compound, train_protein,
                                                     train_mol_embed,train_prot_embed, train_pair_embed)
                            train_loss = LOSS_F(predicts, train_labels.view(-1, 1))
                            train_losses_in_epoch.append(train_loss.item())
                            train_loss.backward()

                            optimizer.step()
                        scheduler.step()
                        train_loss_a_epoch = np.average(train_losses_in_epoch)
                        writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
                        

                        """valid"""
                        valid_pbar = tqdm(
                            enumerate(
                                BackgroundGenerator(valid_dataset_load)),
                            total=len(valid_dataset_load))
                        valid_losses_in_epoch = []
                        model.eval()
                        total_preds = torch.Tensor()
                        total_labels = torch.Tensor()
                        with torch.no_grad():
                            for valid_i, valid_data in valid_pbar:
                                valid_id, valid_compound, valid_protein, valid_mol_embed, valid_prot_embed, valid_pair_embed, valid_labels = valid_data
                                valid_id = valid_id.cuda()
                                valid_compound = valid_compound.cuda()
                                valid_protein = valid_protein.cuda()
                                valid_mol_embed = valid_mol_embed.cuda()
                                valid_prot_embed = valid_prot_embed.cuda()
                                valid_pair_embed = valid_pair_embed.cuda()
                                valid_labels = valid_labels.cuda()

                                valid_predictions = model.forward(valid_id, valid_compound, valid_protein,
                                                                  valid_mol_embed, valid_prot_embed, valid_pair_embed)

                                valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                                valid_losses_in_epoch.append(valid_loss.item())
                                total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                                total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                            Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
                        # Calculate validation metrics
                        valid_MSE = mean_squared_error(Y, P)
                        valid_MAE = mean_absolute_error(Y, P)
                        valid_R2 = r2_score(Y, P)
                        valid_CI = concordance_index(Y, P)
                        valid_RM2 = rm2(Y, P)
                        valid_PS = pearson(Y, P)
                        valid_SM = spearman(Y, P)
                        valid_loss_a_epoch = np.average(valid_losses_in_epoch)  

                        # Save best model based on validation MSE
                        if valid_MSE < valid_best_score:
                            valid_best_score = valid_MSE
                            valid_best_epoch = epoch
                            patience = 0
                            torch.save(model.state_dict(), save_path_i + 'valid_best_checkpoint.pth')
                        else:
                            patience+=1
                        epoch_len = len(str(Epoch))
                        print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                                     f'valid_MSE: {valid_MSE:.5f} ' +
                                     f'valid_MAE: {valid_MAE:.5f} ' +
                                     f'valid_R2: {valid_R2:.5f} ' +
                                     f'valid_CI: {valid_CI:.5f} ' +
                                     f'valid_RM2: {valid_RM2:.5f} ' +
                                     f'valid_PS: {valid_CI:.5f} ' +
                                     f'valid_SM: {valid_RM2:.5f} '+
                                     f'patience: {patience}')
                        print(print_msg)
                        with open(save_path_i + "valid_results.txt", 'a') as f:
                            f.write(print_msg + '\n')
                        writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
                        writer.add_scalar('Valid MSE', valid_MSE, epoch)
                        writer.add_scalar('Valid MAE', valid_MAE, epoch)
                        writer.add_scalar('Valid R2', valid_R2, epoch)
                        writer.add_scalar('Valid CI', valid_CI, epoch)
                        writer.add_scalar('Valid RM2', valid_RM2, epoch)
                        writer.add_scalar('Valid PS', valid_PS, epoch)
                        writer.add_scalar('Valid SM', valid_SM, epoch)


                        
                    torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
                    """load trained model"""
                    model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
                    # Evaluate on train, valid, test sets
                    trainset_test_results, _, _, _, _, _, _, _ = test_model(train_dataset_load, save_path_i, DATASET,
                                                                            lable="Train")
                    validset_test_results, _, _, _, _, _, _, _ = test_model(valid_dataset_load, save_path_i, DATASET,
                                                                            lable="Valid")
                    testset_test_results, mse_test, mae_test, r2_test, ci_test, rm2_test, ps_test, sm_test = test_model(
                        test_dataset_load, save_path_i, DATASET, lable="Test")
                    with open(save_path + "The_results.txt", 'a') as f:
                        # f.write("results on {}th fold\n".format(i_fold+1))
                        f.write(trainset_test_results + '\n')
                        f.write(validset_test_results + '\n')
                        f.write(testset_test_results + '\n')
                    writer.close()
                    # Store metrics for this fold
                    MSE_List.append(mse_test)
                    MAE_List.append(mae_test)
                    R2_List.append(r2_test)
                    CI_List.append(ci_test)
                    RM2_List.append(rm2_test)
                    PS_List.append(ps_test)
                    SM_List.append(sm_test)
                # Calculate mean and std for all folds
                MSE_mean, MSE_var = np.mean(MSE_List), np.sqrt(np.var(MSE_List))
                MAE_mean, MAE_var = np.mean(MAE_List), np.sqrt(np.var(MAE_List))
                R2_mean, R2_var = np.mean(R2_List), np.sqrt(np.var(R2_List))
                CI_mean, CI_var = np.mean(CI_List), np.sqrt(np.var(CI_List))
                RM2_mean, RM2_var = np.mean(RM2_List), np.sqrt(np.var(RM2_List))
                PS_mean, PS_var = np.mean(PS_List), np.sqrt(np.var(PS_List))
                SM_mean, SM_var = np.mean(SM_List), np.sqrt(np.var(SM_List))
                with open(save_path + 'The_results.txt', 'a') as f:
                    f.write('The results on {}:'.format(DATASET) + '\n')
                    f.write('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var) + '\n')
                    f.write('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var) + '\n')
                    f.write('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var) + '\n')
                    f.write('CI(std):{:.4f}({:.4f})'.format(CI_mean, CI_var) + '\n')
                    f.write('RM2(std):{:.4f}({:.4f})'.format(RM2_mean, RM2_var) + '\n')
                    f.write('PS(std):{:.4f}({:.4f})'.format(PS_mean, PS_var) + '\n')
                    f.write('SM(std):{:.4f}({:.4f})'.format(SM_mean, SM_var) + '\n''\n')

                print('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var))
                print('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var))
                print('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var))
                print('CI(std):{:.4f}({:.4f})'.format(CI_mean, CI_var))
                print('RM2(std):{:.4f}({:.4f})'.format(RM2_mean, RM2_var))
                print('PS(std):{:.4f}({:.4f})'.format(PS_mean, PS_var))
                print('SM(std):{:.4f}({:.4f})'.format(SM_mean, SM_var))
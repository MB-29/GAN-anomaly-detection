import numpy as np 
import pickle 
import os 

if __name__ == '__main__':
    
    print()
    print("INITIALISATION :")
    print("================")
    print()
    
    # Paramètres
    path = "results/pickles" # path d'enregistrement
    names = ["_err_1","_rec_1","_norm_1","_norm_rec_1","_plot_1","_plot_rec_1"]
    base_name = "embedding"
    print("Parametres : ")
    print("path :",path)
    print()
    
    # Chargement des données
    print("Chargement des données : ")
    
    current_path = os.getcwd()
    os.chdir(path)
    
    embedding_dic = dict()
    for i in names :
        embedding_dic[base_name+i] = pickle.load(open(base_name+i, 'rb')) 
    
    for i in names:
        print(base_name+i)
        for j in embedding_dic[base_name+i].keys():
            embedding_dic[base_name+i][j] = np.around(embedding_dic[base_name+i][j][0].data.numpy(),5)
            print("image ",j," taille :",embedding_dic[base_name+i][j].shape)
        print()
        
    nb_img = len(list(embedding_dic[base_name+i].keys()))
    os.chdir(current_path)
        
   
    # Première étude 
    print("PREMIERE ETUDE :")
    print("================")
    print()
    
    print("Images initiales :")
    print("(parametres : mean, stdn min, max)")
    img_err = embedding_dic[base_name+names[2]][0]
    print("image ",0,names[2]," initiale     : ",np.around(img_err.mean(),5),",",np.around(img_err.std(),5),",",img_err.min(),",",img_err.max())
    for k in range(nb_img):
        img_err = embedding_dic[base_name+names[4]][k]
        print("image ",k,names[4]," initiale     : ",np.around(img_err.mean(),5),",",np.around(img_err.std(),5),",",img_err.min(),",",img_err.max())
    print()
    
    print("Images reconstruites :")
    print("(parametres : mean, stdn min, max)")
    img_err_rec = embedding_dic[base_name+names[3]][0]
    print("image ",0,names[3]," reconstruite     : ",np.around(img_err_rec.mean(),5),",",np.around(img_err_rec.std(),5),",",img_err_rec.min(),",",img_err_rec.max())
    for k in range(nb_img):
        img_err_rec = embedding_dic[base_name+names[5]][k]
        print("image ",k,names[5]," reconstruite     : ",np.around(img_err_rec.mean(),5),",",np.around(img_err_rec.std(),5),",",img_err_rec.min(),",",img_err_rec.max())
    
    
    # Enregistrer en numpy du embedding_dic pour pouvoir visualiser et étudier sur jupyter
    print()
    print("Enregistrement")
    print("==============")
    current_path = os.getcwd()
    os.chdir(path)
    pickle.dump(embedding_dic, open(str("embedding_all_numpy_2"), 'wb')) # embedding_all_numpy_2 à changer pour sauvegarder 
    os.chdir(current_path)
    
    
    
    
    
    
    
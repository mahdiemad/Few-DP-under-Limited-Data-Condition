import numpy.random as rng
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import combinations, product
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os,json,re

from sys import stdout
def flush(string):
    stdout.write('\r')
    stdout.write(str(string))
    stdout.flush()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,X_train,y_train,X_val,y_val):
        self.data = {'train':X_train,'val':X_val}
        self.labels = {'train':y_train,'val':y_val}
        train_classes = list(set(y_train))
        np.random.seed(10)
#         train_classes = sorted(rng.choice(train_classes,size=(int(len(train_classes)*0.8),),replace=False) )
        self.classes = {'train':sorted(train_classes),'val':sorted(list(set(y_val)))}
        self.indices = {'train':[np.where(y_train == i)[0] for i in self.classes['train']],
                        'val':[np.where(y_val == i)[0] for i in self.classes['val']]
                       }
        print(self.classes)
        print(len(X_train),len(X_val))
        print([len(c) for c in self.indices['train']],[len(c) for c in self.indices['val']])
        
    def set_val(self,X_val,y_val):
        self.data['val'] = X_val
        self.labels['val'] = y_val
        self.classes['val'] =  sorted(list(set(y_val)))
        self.indices['val'] =  [np.where(y_val == i)[0] for i in self.classes['val']]
        
    def set_train(self,X,y):
        self.data['train'] = X
        self.labels['train'] = y
        self.classes['train'] =  sorted(list(set(y)))
        self.indices['train'] =  [np.where(y == i)[0] for i in self.classes['train']]    

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w, h = X.shape
#         if batch_size > n_classes:
#             raise ValueError("{} batch_size has greter than {} classes".format(batch_size,n_classes))

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=True)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, w,h,1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            n_examples = len(X_indices[category])
            if(n_examples==0):
                print("error:n_examples==0",n_examples)
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[X_indices[category][idx_1]].reshape(w, h, 1)
            #pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category  
                idx_2 = (idx_1 + rng.randint(1,n_examples)) % n_examples
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
                n_examples = len(X_indices[category_2])
                idx_2 = rng.randint(0, n_examples)
            pairs[1][i,:,:,:] = X[X_indices[category_2][idx_2]].reshape(w, h,1)
        return pairs, targets, categories
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w, h = X.shape
        if N > n_classes:
            raise ValueError("{} way task has greter than {} classes".format(N,n_classes))

        categories = rng.choice(n_classes,size=(N,),replace=False)            
        true_category = categories[0]
        n_examples = len(X_indices[true_category]) 
        ex1, ex2 = rng.choice(n_examples,size=(2,),replace=False)
        test_image = np.asarray([X[X_indices[true_category][ex1]]]*N).reshape(N, w, h,1)
        support_set = np.zeros((N,w,h))
        support_set[0,:,:] = X[X_indices[true_category][ex2]]
        for idx,category in enumerate(categories[1:]):
            n_examples = len(X_indices[category])
            support_set[idx+1,:,:] = X[X_indices[category][rng.randint(0,n_examples)]]
        support_set = support_set.reshape(N, w, h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, categories)
        pairs = [test_image,support_set]

        return pairs, targets,categories
                                    
    def make_oneshot_task2(self,idx,s="val",support_set=[]):
        """Create pairs_list of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        X_labels = self.labels[s]

        X_train=self.data['train']
        indices_train = self.indices['train']
        classes_train = self.classes['train']
        N = len(indices_train)
        
        _, w,h = X.shape
        
        test_image = np.asarray([X[idx]]*N)

        if(len(support_set) == 0):
            support_set = np.zeros((N,w,h))
            for index in range(N):
                support_set[index,:] = X_train[rng.choice(indices_train[index],size=(1,),replace=False)]
            support_set = support_set

        targets = np.zeros((N,))
        # true_index = classes_train.index(X_labels[X_labels.index[idx]])
        true_index = classes_train.index(X_labels[idx])
        targets[true_index] = 1
        
#         targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, classes_train)
        categories = classes_train
     
        pairs = [test_image,support_set]
        
        return pairs, targets,categories
            
    def test_oneshot2(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        k = len(self.labels[s])
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        probs_all = []
        err_print_num = 0
        for idx in range(k):
            inputs, targets,categories = self.make_oneshot_task2(idx,s)
            n_classes, w, h = inputs[0].shape
            inputs[0]=inputs[0].reshape(n_classes,w,h)
            inputs[1]=inputs[1].reshape(n_classes,w,h)
            probs = -model.predict(inputs,verbose=0)
            # print(probs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
            elif verbose and err_print_num<1:
                err_print_num = err_print_num +1
                print(targets)
#                 print(categories)
                print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                inputs[0]=inputs[0].reshape(n_classes,w,h,1)
                inputs[1]=inputs[1].reshape(n_classes,w,h,1)
    #            plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
            preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
            probs_all.append(probs)
#             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct,np.array(preds),np.array(probs_all)
    
        
    def test_fewshot2(self,model,N,k,s="val",verbose=0,shots=1):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        k = len(self.labels[s])
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        probs_all = []
        err_print_num = 0
        X_train=self.data['train']
        X=self.data[s]
        indices_train = self.indices['train']
        N = len(indices_train) 
        _, w,h = X.shape
       
        all_support_set=np.zeros((shots,N,w,h))
        all_selected_idx=[]
        for index in range(N):
            selected_idx=rng.choice(indices_train[index],size=(shots,),replace=False)
            all_selected_idx.append(selected_idx)
            selected= X_train[rng.choice(indices_train[index],size=(shots,),replace=False)]
            for shot in range(shots):
                all_support_set[shot,index,:]=selected[shot]
                
        preds_few_shot = []
        prods_few_shot = []
        scores = []
        for shot in range(shots):
            n_correct = 0
            preds = []
            probs_all = []
            for idx in range(k):
                inputs, targets,categories = self.make_oneshot_task2(idx,s,all_support_set[shot])
                n_classes, w,h = inputs[0].shape
                probs = -model.predict(inputs,verbose=0)
                if np.argmax(probs) == np.argmax(targets):
                    n_correct+=1
                elif verbose and err_print_num<1:
                    err_print_num = err_print_num +1
                    print(targets)
        #                 print(categories)
                    print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                    plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
                preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                probs_all.append(probs)
        #             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
            percent_correct = (100.0*n_correct / k)
            percent_correct,preds,probs_all=percent_correct,np.array(preds),np.array(probs_all)
            if verbose:
                print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
            print(percent_correct,preds.shape,probs_all.shape)
            scores.append(percent_correct)
            preds_few_shot.append(preds[:,1])
            prods_few_shot.append(probs_all)
            preds = []
            for line in np.array(preds_few_shot).T:
                pass
                preds.append(np.argmax(np.bincount(line)))
        #             utils.confusion_plot(np.array(preds),data.y_test) 
            prod_preds = np.argmax(np.sum(prods_few_shot,axis=0),axis=1).reshape(-1)

            score_few_shot = accuracy_score(self.labels['val'],np.array(preds))*100

            score_few_shot_prob = accuracy_score(self.labels['val'],prod_preds)*100
        print('{}_shot Accuracy based on {} one-shot prediction:'.format(shots,shots),score_few_shot)  
        print('{}_shot Accuracy based on _probabilty:'.format(shots),score_few_shot_prob)
        return score_few_shot,score_few_shot_prob,preds,prod_preds
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),)
 

def generate_pairs(X, y):
    """
    Generates all possible similar and dissimilar pairs from the dataset.
    
    Arguments:
    - X: Array of samples (shape: [N, features]).
    - y: Array of labels (shape: [N]).
    
    Returns:
    - pairs: List of sample pairs (X1, X2).
    - labels: List of corresponding labels (1 for similar, 0 for dissimilar).
    """
    similar_pairs = []
    dissimilar_pairs = []
    
    # Generate similar pairs
    for label in np.unique(y):
        class_samples = X[y == label]
        similar_pairs += list(combinations(class_samples, 2))
    
    # Generate dissimilar pairs
    for label1, label2 in combinations(np.unique(y), 2):
        class1_samples = X[y == label1]
        class2_samples = X[y == label2]
        dissimilar_pairs += list(product(class1_samples, class2_samples))
    
    # Prepare labels
    similar_labels = [1] * len(similar_pairs)
    dissimilar_labels = [0] * len(dissimilar_pairs)
    
    # Combine pairs and labels
    all_pairs = similar_pairs + dissimilar_pairs
    all_labels = similar_labels + dissimilar_labels
    
    return np.array(all_pairs), np.array(all_labels)
 

def train_full(settings,siamese_net,siamese_loader):

    pairs, labels = generate_pairs(siamese_loader.data['train'], siamese_loader.labels['train'])
    
    # Prepare inputs for Siamese network
    pairs_left = pairs[:, 0]  # First element in the pair
    pairs_right = pairs[:, 1]  # Second element in the pair
    labels = labels.astype(np.float32)  # Similarity labels

    early_stopping = EarlyStopping(
    monitor='loss',  # Monitor training loss
    patience=10,     # Stop after 10 epochs with no improvement
    restore_best_weights=True)

    # Define model checkpoint
    model_checkpoint = ModelCheckpoint(
        filepath=settings["save_path"]+'best_siamese_model.h5',
        monitor='loss',
        save_best_only=True,
        verbose=1)

    # Train the model
    history = siamese_net.fit(
        [pairs_left, pairs_right],  # Input pairs
        labels,                     # Corresponding labels
        batch_size=512,              # Batch size
        epochs=settings['n_iter'],                 # Maximum number of epochs
        callbacks=[early_stopping, model_checkpoint],  # Callbacks
        shuffle=True                # Shuffle data
        )

    # Train the model using model.fit
    # siamese_net.fit(
        # [input_1, input_2], 
        # labels, 
        # batch_size=512, 
        # epochs=settings['n_iter'], 
        # validation_split=0.2)
        
    return history
 
def train_and_test_oneshot(settings,siamese_net,siamese_loader):

    settings['best'] = -1 
    settings['n'] = 0
    print(settings)

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print("training...")
    
    #Training loop
    for i in range(settings['n'], settings['n_iter']):
        (inputs,targets,_)=siamese_loader.get_batch(settings['batch_size'])
        n_classes, w, h,_ = inputs[0].shape
        loss, accuracy=siamese_net.train_on_batch(inputs,targets)

        if accuracy >= settings['best'] :
            siamese_net.save(weights_path)
            settings['best'] = accuracy
            settings['n'] = i
            with open(os.path.join(weights_path+".json"), 'w') as f:
                f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': ')))

        if i % settings['loss_every'] == 0:
            flush("{} -> Loss: {:.5f}, Train_ACC: % {:.5f},".format(i,loss,100*accuracy))
            
    return "\n ********* Finished ********* \n best result on Training: % {:.5f}".format(100*settings['best'])
 
 
def _mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            print("can't create directory '{}''".format(path))
            exit(1)
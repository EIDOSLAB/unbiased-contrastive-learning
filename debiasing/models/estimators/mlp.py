from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP


class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, shuffle=True, batch_size='auto', learning_rate='constant', 
                 learning_rate_init=0.001, max_iter=500, n_iter_no_change=50, tol=1e-8, 
                 beta_1=0.9, beta_2=0.999, device='auto', ddp=False, device_id=0):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.loss_curve_ = []
        self.mlp = None
        self.ddp = ddp
        self.device_id = device_id

    def construct(self, sample, n_classes):
        input_dim = sample.shape[1]

        hidden = []
        for hidden_size in self.hidden_layer_sizes:
            hidden.append(nn.Linear(input_dim, hidden_size))
            if self.activation == 'relu':
                hidden.append(nn.ReLU(inplace=True))
            elif self.activation == 'tanh':
                hidden.append(nn.Tanh())
            input_dim = hidden_size
        
        output = nn.Linear(input_dim, n_classes)
        self.mlp = nn.Sequential(*hidden, output).to(self.device)
        self.mlp(sample)

        if self.ddp:
            print("Using DDP")
            self.mlp = DDP(self.mlp, device_ids=[self.device_id], output_device=self.device_id)
            self.device = self.device_id

        if self.solver == 'adam':
            self.optimizer = torch.optim.Adam(self.mlp.parameters(),
                                              lr=self.learning_rate_init,
                                              weight_decay=self.alpha,
                                              betas=(self.beta_1, self.beta_2))

        elif self.solver == 'sgd':
            self.optimizer = torch.optim.SGD(self.mlp.parameters(),
                                             lr=self.learning_rate_init,
                                             weight_decay=self.alpha,
                                             momentum=0.9, nesterov=True)

        self.scheduler = None
        if self.learning_rate == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.max_iter,
                                                                        verbose=False)

    def fit(self, X_train, y_train):
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train.astype(np.int64)).to(self.device)

        if self.mlp is None:
            N_y = len(torch.unique(y_train))
            self.construct(n_classes=N_y, sample=X_train[:2])

        n_iter = 0
        n_iter_no_change = 0
        best_loss = np.inf

        N = len(X_train)
        batch_size = min(200, N) if self.batch_size == 'auto' else self.batch_size

        while n_iter < self.max_iter:
            X_shuffle = X_train
            y_shuffle = y_train

            if self.shuffle:
                perm = torch.randperm(N)
                X_shuffle = X_shuffle[perm]
                y_shuffle = y_shuffle[perm]
            
            start_idx = 0
            running_loss = 0.
            batch_idx = 1

            while (start_idx+batch_size) <= N:
                X_batch = X_shuffle[start_idx:start_idx+batch_size]
                y_batch = y_shuffle[start_idx:start_idx+batch_size]
                
                with torch.enable_grad():
                    y_pred = self.mlp(X_batch)
                    loss = F.cross_entropy(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                start_idx += batch_size
                running_loss += loss.item()
                batch_idx += 1

            running_loss /= N

            delta = best_loss - running_loss
            # print('Loss:', running_loss, 'best:', best_loss, 'delta:', delta)

            if delta < self.tol:
                n_iter_no_change += 1
            else:
                n_iter_no_change = 0
            
            if n_iter_no_change > self.n_iter_no_change:
                break
            
            if running_loss < best_loss:
                best_loss = running_loss
            
            self.loss_curve_.append(running_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            n_iter += 1

        if n_iter >= self.max_iter:
            print('MLPClassifier ConvergenceWarning: max_iter reached before convergence')
        self.n_iter_ = n_iter
        self.print_info()

    def print_info(self):
        print(f'MLPClassifier: trained for {self.n_iter_} iterations')
        print(f'               final loss: {self.loss_curve_[-1]}')

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            preds = self.mlp(X)
            _, y = torch.max(preds, dim=1)
        return y.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        return accuracy_score(y, y_pred)

    def __repr__(self):
        return f"MLPClassifier(solver={self.solver}, hidden_layer_sizes={self.hidden_layer_sizes})"


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.neural_network import MLPClassifier as MLPClassifier_sk
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000, n_features=128, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    names = ['sklearn-mlp', 'pytorch-mlp', 'pytorch-fc']
    classifiers = [MLPClassifier_sk(solver='sgd'), 
                   MLPClassifier(solver='sgd', device='auto'),
                   MLPClassifier(hidden_layer_sizes=(), solver='sgd', device='auto')]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('estimator', name, clf, 'score:', score, clf.n_iter_, clf.loss_curve_[-1])

    print('Running MNIST')

    from torchvision import datasets
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std of the MNIST dataset
    ])
    
    train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('.', train=False, transform=transform)

    X_train = train_data.data.numpy()
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train = train_data.targets.numpy()

    X_test = test_data.data.numpy()
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test = test_data.targets.numpy()

    print('Dataset size:', X_train.shape, X_test.shape)
    
    print('Using solver Adam')
    names = ['sklearn-mlp', 'pytorch-mlp']
    classifiers = [MLPClassifier_sk(hidden_layer_sizes=(256, 256,), solver='adam'), 
                   MLPClassifier(hidden_layer_sizes=(256, 256,), solver='adam', device='auto')]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('estimator', name, 'score:', score, clf.n_iter_, clf.loss_curve_[-1])

    print('Using solver SGD')
    names = ['sklearn-mlp', 'pytorch-mlp']
    classifiers = [MLPClassifier_sk(hidden_layer_sizes=(256, 256,), solver='sgd'), 
                   MLPClassifier(hidden_layer_sizes=(256, 256,), solver='sgd', device='auto')]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('estimator', name, 'score:', score, clf.n_iter_, clf.loss_curve_[-1])


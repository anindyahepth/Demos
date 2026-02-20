import mlflow
import torch
import torch.nn as nn
from sam import SAM
from torch.optim import SGD, AdamW
from utils.utils import get_datasets, get_dataloaders
from utils.utils import get_model_resnet
from utils.utils import train_single_epoch, inference


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = 100

    # Dictionary to store results
    results = {}
    trainable_params = 0

    # Define optimizers to compare
    optimizers_to_test = {
    'SAM-AdamW': {
        'factory': lambda model: SAM(
            params=model.parameters(),
            base_optimizer=AdamW(model.parameters(), lr=1e-4, weight_decay=0.05),
            rho=0.05,
            p=2
        ),
        'lr': 1e-4,
        'rho': 0.05,
        'p': 2,
        'optimizer_class': 'SAM',
        'base_optimizer': 'AdamW'
    },
    'AdamW': {
        'factory': lambda model: AdamW(model.parameters(), lr=1e-4, weight_decay=0.05),
        'lr': 1e-4,
        'optimizer_class': 'AdamW'
    },
    'SGD': {
       'factory': lambda model: SGD(
           model.parameters(),
           lr=0.05,
           momentum=0.0,      # Standard momentum 0.9
           weight_decay=0.0  # L2 regularization  5e-4
       ),
       'lr': 0.05,
       'momentum': 0.0,
       'weight_decay': 0.0,
       'optimizer_class': 'SGD'
   },
    'SAM-SGD': {
            'factory': lambda model: SAM(
                params=model.parameters(),
                base_optimizer=SGD(
                    model.parameters(),
                    lr=0.05,
                    momentum=0.0,
                    weight_decay=0.0
                ),
                rho=0.05,
                p=2
            ),
            'lr': 0.05,
            'momentum': 0.0,
            'weight_decay': 0.0,
            'rho': 0.05,
            'p': 2,
            'optimizer_class': 'SAM',
            'base_optimizer': 'SGD'
        }
}

    #Datasets and Dataloaders
    dataset_train, dataset_test = get_datasets(augment=True)
    train_loader, test_loader = get_dataloaders(dataset_train, dataset_test)

    # Train with each optimizer
    for opt_name, opt_config in optimizers_to_test.items():
        print(f"\n{'='*60}")
        print(f"Training with {opt_name}")
        print(f"{'='*60}\n")

        with mlflow.start_run(experiment_id=0):
            # Set tags for easier filtering
            mlflow.set_tag("optimizer", opt_name)
            mlflow.set_tag("dataset", "CIFAR10")
            mlflow.set_tag("model_type", "ResNet")

            # Initialize fresh model
            model = get_model_resnet().to(device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters:{trainable_params/1e6} M")

            optimizer = opt_config['factory'](model)

            # Detect if optimizer is SAM
            use_sam = hasattr(optimizer, 'first_step') and hasattr(optimizer, 'second_step')
            record_frequency = 1 if use_sam else 2

            # Adjust number of epochs for fair comparison
            actual_n_epochs = n_epochs if use_sam else 2 * n_epochs

            # Log hyperparameters
            params = {
                "optimizer_name": opt_name,
                "optimizer_class": opt_config['optimizer_class'],
                "learning_rate": opt_config['lr'],
                "n_epochs": actual_n_epochs,
                "base_n_epochs": n_epochs,
                "batch_size": train_loader.batch_size,
                "model_architecture": "ViT",
                "input_size": 3*32*32,
                "output_size": 10,
                "device": str(device),
                "use_sam": use_sam,
                "record_frequency": record_frequency,
                "criterion": "CrossEntropyLoss",
                "train_dataset_size": len(train_loader.dataset),
                "test_dataset_size": len(test_loader.dataset)
            }

            # Add base optimizer for SAM
            if 'base_optimizer' in opt_config:
                params['base_optimizer'] = opt_config['base_optimizer']

            mlflow.log_params(params)

            # Lists to store metrics
            train_loss_list = []
            train_acc_list = []
            train_loss_list_st = []
            train_acc_list_st = []
            test_acc_list =[]
            test_acc_list_st =[]


            # Train
            for epoch in range(actual_n_epochs):
                train_loss, train_acc = train_single_epoch(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    optimizer=optimizer,
                    epoch=epoch,
                    criterion=nn.CrossEntropyLoss()
                )
                test_acc = inference(model, test_loader, device)



                # Store in Lists
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                # Calculate generalization gap
                gen_gap = train_acc - test_acc

                # Log metrics to MLflow (every epoch)
                metrics_epoch = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "generalization_gap": gen_gap
                }
                mlflow.log_metrics(metrics_epoch, step=epoch)

                # Record standardized metrics
                if (epoch + 1) % record_frequency == 0:
                    train_loss_list_st.append(train_loss)
                    train_acc_list_st.append(train_acc)
                    test_acc_list_st.append(test_acc)

                    # Log standardized metrics (for fair comparison)
                    step_st = len(train_loss_list_st) - 1
                    metrics_std = {
                        "train_loss_st": train_loss,
                        "train_accuracy_st": train_acc,
                        "test_accuracy_st": test_acc,
                        "generalization_gap_st": gen_gap
                    }
                    mlflow.log_metrics(metrics_std, step=step_st)


            # Calculate and log final/summary metrics
            best_train_acc = max(train_acc_list)
            best_train_epoch = train_acc_list.index(best_train_acc)
            worst_train_loss = max(train_loss_list)
            best_train_loss = min(train_loss_list)
            best_test_acc = max(test_acc_list)

            final_metrics = {
                "final_train_loss": train_loss_list[-1],
                "final_train_accuracy": train_acc_list[-1],
                "best_train_accuracy": best_train_acc,
                "best_train_epoch": best_train_epoch,
                "best_train_loss": best_train_loss,
                "worst_train_loss": worst_train_loss,
                "avg_train_loss": sum(train_loss_list) / len(train_loss_list),
                "final_test_accuracy": test_acc_list[-1],
                "best_test_accuracy": best_test_acc
            }
            mlflow.log_metrics(final_metrics)

            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=f"CIFAR10_{opt_name}_model"
            )

            # Store results
            results[opt_name] = {
                'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'test_acc': test_acc_list,
                'train_loss_st': train_loss_list_st,
                'train_acc_st': train_acc_list_st,
                'test_acc_st': test_acc_list_st,
                'generalization_gaps': [train - test for train, test in zip(train_acc_list, test_acc_list)],
                'generalization_gaps_st': [train - test for train, test in zip(train_acc_list_st, test_acc_list_st)],
                'mlflow_run_id': mlflow.active_run().info.run_id
            }

            print(f"\nFinal results for {opt_name}:")
            print(f"  Train Loss: {train_loss_list[-1]:.4f}")
            print(f"  Train Accuracy: {train_acc_list[-1]:.4f}")
            print(f"  Best train Accuracy: {best_train_acc:.4f} (epoch {best_train_epoch})")
            print(f" Final Test Accuracy: {test_acc_list[-1]:.4f} ")
            print(f"  MLflow Run ID: {mlflow.active_run().info.run_id}")

    print("\n" + "="*60)
    print("All experiments logged to MLflow!")
    print("View results with: mlflow ui")
    print("="*60)

import os
import sys
import numpy as np
from copy import deepcopy

import models
from constant import *
from clients.MFCL import MFCL_client
from models.ResNet import ResNet18
from models.myNetwork import network
from data_prep.data import CL_dataset
from clients.simple import AVG, PROX, ORACLE
from data_prep.super_imagenet import SuperImageNet
from utiles import setup_seed, fedavg_aggregation, evaluate_accuracy_forgetting, evaluate_accuracy, train_gen, start
from logger import (setup_logger, log_task_results, log_forgetting_results, 
                   log_final_results, save_experiment_config, save_results_summary,
                   log_error, validate_dataset_path)


def main():
    args = start()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
    setup_seed(args.seed)

    # Setup logger and output directory
    logger, log_dir = setup_logger(args)
    save_experiment_config(args, log_dir)

    # Store experiment results for cleanup on error
    max_accuracy = []
    all_task_accuracies = []

    try:
        logger.info("Validating dataset configuration...")
        
        # Validate dataset path if provided
        if hasattr(args, 'path') and args.path:
            if not validate_dataset_path(logger, args.dataset, args.path):
                raise FileNotFoundError(f"Dataset validation failed for {args.dataset} at {args.path}")
        elif args.dataset != CIFAR100:
            log_error(logger, f"Dataset path is required for {args.dataset}. Use --path argument.")
            raise ValueError(f"Missing dataset path for {args.dataset}")

        logger.info("Initializing dataset and model...")

        if args.dataset == CIFAR100:
            dataset = CL_dataset(args)
            feature_extractor = ResNet18(args.num_classes, cifar=True)
            ds = dataset.train_dataset
            logger.info("CIFAR100 dataset initialized")
            
        elif args.dataset == tinyImageNet:
            dataset = CL_dataset(args)
            feature_extractor = ResNet18(args.num_classes, cifar=False)
            ds = dataset.train_dataset
            args.generator_model = 'TINYIMNET_GEN'
            logger.info("Tiny ImageNet dataset initialized")
            
        elif args.dataset == SuperImageNet:
            from models.imagenet_resnet import resnet18
            dataset = SuperImageNet(args.path, version=args.version, num_tasks=args.n_tasks, 
                                   num_clients=args.num_clients, batch_size=args.batch_size)
            args.num_classes = dataset.num_classes
            feature_extractor = resnet18(args.num_classes)
            args.generator_model = 'IMNET_GEN'
            args.img_size = dataset.img_size
            ds = dataset
            logger.info("SuperImageNet dataset initialized")
            
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        global_model = network(dataset.n_classes_per_task, feature_extractor)
        teacher, generator = None, None
        gamma = np.log(args.lr_end / args.lr)
        task_size = dataset.n_classes_per_task
        counter, classes_learned = 0, task_size
        num_participants = int(args.frac * args.num_clients)
        clients = []

        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Number of classes per task: {task_size}")
        logger.info(f"Number of participants per round: {num_participants}")

        if args.method == MFCL:
            try:
                generator = models.__dict__['generator'].__dict__[args.generator_model](zdim=args.z_dim, convdim=args.conv_dim)
                logger.info("Generator model initialized for MFCL")
            except KeyError as e:
                log_error(logger, f"Generator model '{args.generator_model}' not found", e)
                raise

        logger.info("Initializing clients...")
        for i in range(args.num_clients):
            group = dataset.groups[i]
            if args.method == FedAVG:
                client = AVG(args.batch_size, args.epochs, ds, group, args.dataset)
            elif args.method == FedProx:
                client = PROX(args.batch_size, args.epochs, ds, group, args.dataset)
            elif args.method == ORACLE:
                client = ORACLE(args.batch_size, args.epochs, ds, group, args.dataset)
            elif args.method == MFCL:
                client = MFCL_client(args.batch_size, args.epochs, ds, group, args.w_kd, args.w_ft, args.syn_size, args.dataset)
            clients.append(client)

        logger.info(f"Initialized {len(clients)} clients")
        logger.info("Starting federated continual learning...")

        for t in range(args.n_tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING TASK {t}")
            logger.info(f"{'='*60}")
            
            try:
                test_loader = dataset.get_full_test(t)
                [client.set_next_t() for client in clients]
                
                logger.info(f"Task {t}: Training for {args.global_round} rounds...")
                
                for round in range(args.global_round):
                    weights = []
                    lr = args.lr * np.exp(round / args.global_round * gamma)
                    selected_clients = [clients[idx] for idx in np.random.choice(args.num_clients, num_participants, replace=False)]
                    
                    for user in selected_clients:
                        model = deepcopy(global_model)
                        user.train(model, lr, teacher, generator, counter)
                        weights.append(model.state_dict())
                    
                    global_model.load_state_dict(fedavg_aggregation(weights))
                    
                    if (round + 1) % args.eval_int == 0:
                        correct, total = evaluate_accuracy(global_model, test_loader, args.method)
                        accuracy = 100 * correct / total
                        log_task_results(logger, t, round + 1, accuracy, args.method)
                    
                    counter += 1
                
                # Record maximum accuracy for this task
                if t == 0:
                    max_accuracy.append(correct / total)
                    all_task_accuracies.append([100 * correct / total])
                
                if t > 0:
                    correct, total, accuracies = evaluate_accuracy_forgetting(global_model, dataset.get_cl_test(t), args.method)
                    all_task_accuracies.append(accuracies)
                    max_accuracy.append(accuracies[-1] / 100)  # Convert back to ratio for consistency
                    
                    # Log forgetting results
                    if len(all_task_accuracies) > 1:
                        current_forgetting = sum([all_task_accuracies[0][i] - accuracies[i] for i in range(len(accuracies) - 1)]) / (len(accuracies) - 1) if len(accuracies) > 1 else 0
                    else:
                        current_forgetting = 0
                        
                    log_forgetting_results(logger, t, accuracies, current_forgetting)
                
                # Generator training for next task (except for last task)
                if t != args.n_tasks - 1:
                    if args.method == MFCL:
                        logger.info(f"Training generator for task {t}...")
                        original_global = deepcopy(global_model)
                        teacher = train_gen(deepcopy(global_model), classes_learned, generator, args)
                        for client in clients:
                            client.last_valid_dim = classes_learned
                            client.valid_dim = classes_learned + task_size
                        global_model = original_global
                        logger.info("Generator training completed")
                    
                    classes_learned += task_size
                    global_model.Incremental_learning(classes_learned)
                    logger.info(f"Model expanded to {classes_learned} classes")
                    
            except Exception as e:
                log_error(logger, f"Error during task {t}", e)
                logger.info(f"Stopping experiment at task {t} due to error")
                break

        # Calculate final results
        final_accuracies = all_task_accuracies[-1] if all_task_accuracies else []
        max_accuracy_percentages = [acc * 100 for acc in max_accuracy]  # Convert to percentages

        # Calculate and log final forgetting
        if len(max_accuracy_percentages) > 0 and len(final_accuracies) > 0:
            valid_tasks = min(len(max_accuracy_percentages), len(final_accuracies))
            total_forgetting = sum([max_accuracy_percentages[i] - final_accuracies[i] for i in range(valid_tasks)]) / valid_tasks
            logger.info(f'\nFinal average forgetting: {total_forgetting:.2f}%')

        # Log comprehensive final results
        log_final_results(logger, max_accuracy_percentages, final_accuracies, args.n_tasks)

        # Save results to files
        save_results_summary(log_dir, max_accuracy_percentages, final_accuracies, args.n_tasks, args.method)

        logger.info(f"\nExperiment completed successfully! All results saved to: {log_dir}")
        logger.info("Files generated:")
        logger.info(f"  - experiment.log: Detailed experiment log")
        logger.info(f"  - config.txt: Experiment configuration")
        logger.info(f"  - results_summary.csv: Results in CSV format")
        logger.info(f"  - results_summary.txt: Results summary")

    except FileNotFoundError as e:
        log_error(logger, f"File not found: {str(e)}", e)
        logger.info("Please check your dataset path and ensure the dataset is properly downloaded and extracted.")
        return 1
        
    except Exception as e:
        log_error(logger, f"Unexpected error during experiment: {str(e)}", e)
        
        # Still try to save partial results
        try:
            final_accuracies = all_task_accuracies[-1] if all_task_accuracies else []
            max_accuracy_percentages = [acc * 100 for acc in max_accuracy]
            save_results_summary(log_dir, max_accuracy_percentages, final_accuracies, args.n_tasks, args.method)
            logger.info(f"Partial results saved to: {log_dir}")
        except:
            logger.error("Could not save partial results")
        
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
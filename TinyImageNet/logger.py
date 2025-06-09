import os
import logging
import datetime
import traceback
from pathlib import Path


def setup_logger(args):
    """
    Setup logger with file and console handlers
    
    Args:
        args: argument parser object containing experiment parameters
        
    Returns:
        logger: configured logger object
        log_dir: path to log directory
    """
    # Create output directory structure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name based on key parameters
    exp_name = f"{args.method}_{args.dataset}_clients{args.num_clients}_tasks{args.n_tasks}_seed{args.seed}_{timestamp}"
    
    # Create output directory
    output_dir = Path("outputs")
    log_dir = output_dir / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('MFCL')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler for detailed logs
    log_file = log_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for simple output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log experiment configuration
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    
    # Log all arguments
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    
    logger.info("="*80)
    logger.info(f"Experiment directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    return logger, log_dir


def log_error(logger, error_msg, exception=None):
    """Log error messages with optional exception details"""
    logger.error(f"ERROR: {error_msg}")
    if exception:
        logger.error(f"Exception details: {str(exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


def log_task_results(logger, task_id, round_num, accuracy, method):
    """Log results for a specific task and round"""
    logger.info(f"Task {task_id} | Round {round_num} | Method: {method} | Accuracy: {accuracy:.2f}%")


def log_forgetting_results(logger, task_id, accuracies, avg_forgetting):
    """Log forgetting analysis results"""
    logger.info("-"*60)
    logger.info(f"TASK {task_id} COMPLETION - FORGETTING ANALYSIS")
    logger.info("-"*60)
    
    for i, acc in enumerate(accuracies):
        logger.info(f"Task {i} accuracy: {acc:.2f}%")
    
    logger.info(f"Average forgetting: {avg_forgetting:.2f}%")
    logger.info("-"*60)


def log_final_results(logger, max_accuracy, final_accuracies, n_tasks):
    """Log final experiment results"""
    logger.info("="*80)
    logger.info("FINAL EXPERIMENT RESULTS")
    logger.info("="*80)
    
    if len(final_accuracies) > 0:
        total_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(min(len(max_accuracy), len(final_accuracies)))]) / n_tasks
        
        logger.info("Maximum accuracies achieved per task:")
        for i, max_acc in enumerate(max_accuracy):
            logger.info(f"Task {i}: {max_acc:.2f}%")
        
        logger.info("\nFinal accuracies per task:")
        for i, final_acc in enumerate(final_accuracies):
            logger.info(f"Task {i}: {final_acc:.2f}%")
        
        logger.info(f"\nOverall average forgetting: {total_forgetting:.2f}%")
    else:
        logger.info("No final accuracies available due to experiment termination.")
    
    logger.info("="*80)


def save_experiment_config(args, log_dir):
    """Save experiment configuration to a file"""
    config_file = log_dir / "config.txt"
    
    with open(config_file, 'w') as f:
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("="*50 + "\n")
        
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
        
        f.write("="*50 + "\n")


def save_results_summary(log_dir, max_accuracy, final_accuracies, n_tasks, method):
    """Save results summary to CSV and text files"""
    import csv
    
    # Save to CSV
    csv_file = log_dir / "results_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Max_Accuracy', 'Final_Accuracy', 'Forgetting'])
        
        for i in range(n_tasks):
            if i < len(max_accuracy) and i < len(final_accuracies):
                forgetting = max_accuracy[i] - final_accuracies[i]
                writer.writerow([i, f"{max_accuracy[i]:.2f}", 
                               f"{final_accuracies[i]:.2f}", 
                               f"{forgetting:.2f}"])
            elif i < len(max_accuracy):
                writer.writerow([i, f"{max_accuracy[i]:.2f}", "N/A", "N/A"])
            else:
                writer.writerow([i, "N/A", "N/A", "N/A"])
        
        # Add average forgetting if data is available
        if len(max_accuracy) > 0 and len(final_accuracies) > 0:
            valid_tasks = min(len(max_accuracy), len(final_accuracies))
            avg_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(valid_tasks)]) / valid_tasks
            writer.writerow(['Average', '', '', f"{avg_forgetting:.2f}"])
    
    # Save to text file
    summary_file = log_dir / "results_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Method: {method}\n")
        f.write(f"Number of tasks: {n_tasks}\n\n")
        
        f.write("Task-wise Results:\n")
        f.write("-" * 40 + "\n")
        
        for i in range(n_tasks):
            f.write(f"Task {i}:\n")
            if i < len(max_accuracy):
                f.write(f"  Max Accuracy: {max_accuracy[i]:.2f}%\n")
            else:
                f.write(f"  Max Accuracy: N/A\n")
                
            if i < len(final_accuracies):
                f.write(f"  Final Accuracy: {final_accuracies[i]:.2f}%\n")
                f.write(f"  Forgetting: {max_accuracy[i] - final_accuracies[i]:.2f}%\n")
            else:
                f.write(f"  Final Accuracy: N/A\n")
                f.write(f"  Forgetting: N/A\n")
            f.write("\n")
        
        if len(max_accuracy) > 0 and len(final_accuracies) > 0:
            valid_tasks = min(len(max_accuracy), len(final_accuracies))
            avg_forgetting = sum([max_accuracy[i] - final_accuracies[i] for i in range(valid_tasks)]) / valid_tasks
            f.write(f"Average Forgetting: {avg_forgetting:.2f}%\n")
        else:
            f.write("Average Forgetting: N/A (experiment incomplete)\n")


def validate_dataset_path(logger, dataset_name, dataset_path):
    """Validate that the dataset path exists and has the expected structure"""
    
    logger.info(f"Validating dataset path for {dataset_name}...")
    
    if not os.path.exists(dataset_path):
        log_error(logger, f"Dataset path does not exist: {dataset_path}")
        return False
    
    if dataset_name == "tiny_imagenet":
        expected_paths = [
            os.path.join(dataset_path, "tiny-imagenet-200"),
            os.path.join(dataset_path, "tiny-imagenet-200", "train"),
            os.path.join(dataset_path, "tiny-imagenet-200", "val"),
            os.path.join(dataset_path, "tiny-imagenet-200", "val", "val_annotations.txt")
        ]
        
        missing_paths = []
        for path in expected_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            log_error(logger, f"Missing required paths for Tiny ImageNet:")
            for path in missing_paths:
                logger.error(f"  - {path}")
            
            logger.info("\nTo download and setup Tiny ImageNet dataset:")
            logger.info("1. Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            logger.info("2. Extract to your data directory")
            logger.info("3. Ensure the structure looks like:")
            logger.info("   data/")
            logger.info("   └── tiny-imagenet-200/")
            logger.info("       ├── train/")
            logger.info("       │   ├── n01443537/")
            logger.info("       │   ├── n01629819/")
            logger.info("       │   └── ...")
            logger.info("       └── val/")
            logger.info("           ├── images/")
            logger.info("           └── val_annotations.txt")
            
            return False
        
        # Check if train directory has subdirectories (class folders)
        train_path = os.path.join(dataset_path, "tiny-imagenet-200", "train")
        class_folders = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        
        if len(class_folders) == 0:
            log_error(logger, f"No class folders found in {train_path}")
            return False
        
        logger.info(f"Found {len(class_folders)} classes in training set")
        
    elif dataset_name == "CIFAR100":
        # CIFAR100 will be downloaded automatically by torchvision
        logger.info("CIFAR100 dataset will be downloaded automatically if not present")
        
    elif dataset_name == "super_imagenet":
        # Check for ImageNet structure
        expected_paths = [
            os.path.join(dataset_path, "train"),
            os.path.join(dataset_path, "val")
        ]
        
        missing_paths = []
        for path in expected_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            log_error(logger, f"Missing required paths for SuperImageNet:")
            for path in missing_paths:
                logger.error(f"  - {path}")
            return False
    
    logger.info("Dataset validation successful!")
    return True